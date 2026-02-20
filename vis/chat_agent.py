"""Chat agent for the 3D scene visualization using Langchain.

This module provides a Langchain-based agent that can answer questions about
objects in a 3D scene, query maps using CLIP or LLM, and analyze spatio-temporal
relationships using arcs.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from hydra.utils import instantiate
from langchain.agents import create_agent
from langchain.tools import tool
from omegaconf import DictConfig

try:
    from .chat_history import ChatManager
except ImportError:  # Support direct script-style imports
    from chat_history import ChatManager

if TYPE_CHECKING:
    try:
        from .vis_with_viser import ViserCallbackManager
    except ImportError:
        from vis_with_viser import ViserCallbackManager

log = logging.getLogger(__name__)


# Default system prompt for the chat agent
AGENT_SYSTEM_PROMPT = """You are a helpful assistant for running spatio-temporal analysis of a semantic scene graph.
You help users understand and query objects in the scene, as well as run inference on the relationships between objects, represented by arcs.
You should use user-provided context to figure out the meaning of arcs, for instance, if a user talks about tasks being blocked, you should assume arcs represent tasks.

You have access to tools that let you:
# - Query objects using semantic CLIP features (fast, good for visual/category queries)
- Query objects using LLM reasoning (slower, good for complex/abstract queries)
- Query the scene at a specific time to understand spatio-temporal relationships via arcs
- Identify bottleneck objects and arcs that are frequently blocking other operations

When a user asks about objects in the scene:
# 1. For visual/category queries (e.g., "find red objects", "show me chairs"), use run_clip_query_on_map
1. For abstract/complex queries (e.g., "something to sit on for a meeting"), use run_text_query_on_map
3. For temporal queries (e.g., "what will be happening at Day 2, Hour 10?"), use run_arc_query_on_map
4. For arc-level bottleneck analysis (which arcs are recurring blockers), use identify_task_bottlenecks
5. For object-level bottleneck analysis (which objects are blocking the most), use identify_object_bottlenecks

Always provide clear, concise responses about what you found and why.
"""


@dataclass
class AgentConfig:
    """Configuration for the chat agent."""
    llm_model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = AGENT_SYSTEM_PROMPT


class ChatAgent:
    """Langchain-based agent for handling chat in the 3D scene visualizer."""

    def __init__(
        self,
        manager: "ViserCallbackManager",
        cfg: DictConfig | None = None,
        get_current_timestep: Callable[[], tuple[int, float]] | None = None,
        chat_manager: ChatManager | None = None,
    ):
        """Initialize the chat agent.

        Args:
            manager: The ViserCallbackManager instance for accessing scene data.
            cfg: Hydra configuration for the agent. If None, uses defaults.
            get_current_timestep: Callable that returns (current_day, current_hour) from the UI slider.
            chat_manager: ChatManager instance for maintaining conversation history.
        """
        self.manager = manager
        self.cfg = cfg
        self.get_current_timestep = get_current_timestep
        self.chat_manager = chat_manager or ChatManager()

        # Build the agent
        self._agent_executor = self._build_agent()

    def _get_llm(self):
        """Get the LLM instance from config or defaults."""
        if self.cfg is not None and "llm" in self.cfg:
            return instantiate(self.cfg.llm)
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                max_tokens=1024,
            )

    def _get_system_prompt(self) -> str:
        """Get the system prompt from config or defaults."""
        if self.cfg is not None and "system_prompt" in self.cfg:
            return self.cfg.system_prompt
        return AGENT_SYSTEM_PROMPT

    def _parse_time_query(self, time_for_query: str | None) -> tuple[int, float]:
        """Parse a time query string into (day, hour) tuple."""
        if time_for_query is None:
            if self.get_current_timestep is not None:
                return self.get_current_timestep()
            else:
                return (1, 0.0)

        import re
        match = re.match(r"Day\s*(\d+),?\s*Hour\s*([\d.]+)", time_for_query, re.IGNORECASE)
        if match:
            day = int(match.group(1))
            hour = float(match.group(2))
            return (day, hour)

        match = re.match(r"Day\s*(\d+)", time_for_query, re.IGNORECASE)
        if match:
            day = int(match.group(1))
            return (day, 0.0)

        log.warning("Could not parse time query '%s', using current timestep", time_for_query)
        if self.get_current_timestep is not None:
            return self.get_current_timestep()
        return (1, 0.0)

    def _find_arc_state_index_for_time(self, day: int, hour: float) -> int | None:
        """Find the arc state index that matches the given day and hour."""
        if not self.manager.arc_states:
            return None

        for i, state in enumerate(self.manager.arc_states):
            state_day = state.get("current_day", 0)
            state_hour = state.get("current_hour", 0)
            if state_day == day and abs(state_hour - hour) < 0.5:
                return i

        best_idx = 0
        best_diff = float("inf")
        for i, state in enumerate(self.manager.arc_states):
            state_day = state.get("current_day", 0)
            state_hour = state.get("current_hour", 0)
            state_total = (state_day - 1) * 24 + state_hour
            target_total = (day - 1) * 24 + hour
            diff = abs(state_total - target_total)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        return best_idx

    def _create_tools(self) -> list:
        """Create the Langchain tools for the agent."""
        agent = self
        manager = self.manager

        @tool
        def run_text_query_on_map(query: str) -> str:
            """Query objects in the 3D scene using LLM reasoning."""
            if manager.llm_client is None:
                return "Error: LLM client is not available."

            try:
                result = manager.llm_query(query)
                if result:
                    return f"LLM query for '{query}' found: {result}"
                else:
                    return f"LLM query for '{query}' found no matching objects."
            except Exception as e:
                return f"Error running LLM query: {e}"

        @tool
        def run_arc_query_on_map(query: str, time_for_query: str = "") -> str:
            """Query spatio-temporal relationships (arcs) in the scene at a specific time."""
            time_param = time_for_query if time_for_query else None
            day, hour = agent._parse_time_query(time_param)
            state_idx = agent._find_arc_state_index_for_time(day, hour)

            if state_idx is None or not manager.arc_states:
                return f"No arc data available for Day {day}, Hour {hour}."

            state = manager.arc_states[state_idx]
            arcs = state.get("arcs", [])

            if not arcs:
                return f"No arcs/interactions found at Day {day}, Hour {hour}."

            forward_arcs = [a for a in arcs if a.get("arc_type") == "forward"]
            dependency_arcs = [a for a in arcs if a.get("arc_type") == "dependency"]

            response_parts = [f"At Day {day}, Hour {hour:.1f}:"]

            if forward_arcs:
                response_parts.append(f"\n**Active Interactions ({len(forward_arcs)}):**")
                for arc in forward_arcs[:5]:
                    source = arc.get("source", "unknown")
                    target = arc.get("target", "unknown")
                    label = arc.get("label", "interacts with")
                    response_parts.append(f"  - {source} → {target}: {label}")

            if dependency_arcs:
                response_parts.append(f"\n**Blocked Tasks ({len(dependency_arcs)}):**")
                for arc in dependency_arcs[:5]:
                    source = arc.get("source", "unknown")
                    target = arc.get("target", "unknown")
                    blocked_by = arc.get("blocked_by", "previous task")
                    response_parts.append(f"  - {source} → {target}: blocked by {blocked_by}")

            if "block" in query.lower() or "depend" in query.lower():
                if dependency_arcs:
                    response_parts.append(f"\n{len(dependency_arcs)} tasks are currently blocked waiting for dependencies.")
                else:
                    response_parts.append("\nNo tasks are currently blocked.")

            return "\n".join(response_parts)

        @tool
        def identify_task_bottlenecks(time_for_query: str = "") -> str:
            """Identify tasks (forward arcs) that are recurring blockers for other operations."""
            time_param = time_for_query if time_for_query else None

            if not manager.arc_states:
                return "No arc data available for bottleneck analysis."

            day, hour = agent._parse_time_query(time_param)
            start_idx = agent._find_arc_state_index_for_time(day, hour)

            if start_idx is None:
                start_idx = 0

            all_dependency_arcs = []
            for i in range(start_idx, len(manager.arc_states)):
                state = manager.arc_states[i]
                arcs = state.get("arcs", [])
                for arc in arcs:
                    if arc.get("arc_type") == "dependency":
                        all_dependency_arcs.append({
                            **arc,
                            "state_day": state.get("current_day", 0),
                            "state_hour": state.get("current_hour", 0),
                        })

            if not all_dependency_arcs:
                return f"No dependency arcs found from Day {day}, Hour {hour} onwards. No bottlenecks detected."

            bottleneck_counter: Counter[str] = Counter()
            blocking_details: dict[str, list[dict]] = {}

            for arc in all_dependency_arcs:
                blocked_by = arc.get("blocked_by", "")
                blocked_source = arc.get("source", "unknown")
                blocked_target = arc.get("target", "unknown")

                if blocked_by:
                    bottleneck_counter[blocked_by] += 1
                    if blocked_by not in blocking_details:
                        blocking_details[blocked_by] = []
                    blocking_details[blocked_by].append({
                        "blocked": f"{blocked_source} → {blocked_target}",
                        "day": arc.get("state_day", 0),
                        "hour": arc.get("state_hour", 0),
                    })

            if not bottleneck_counter:
                return "Could not identify bottleneck tasks from dependency arcs."

            top_3 = bottleneck_counter.most_common(3)

            response_parts = [f"**Top 3 Bottleneck Tasks** (from Day {day}, Hour {hour} onwards):\n"]

            for rank, (task, count) in enumerate(top_3, 1):
                response_parts.append(f"{rank}. **{task}** - blocks {count} tasks")

                details = blocking_details.get(task, [])[:3]
                for detail in details:
                    response_parts.append(f"   - Blocks: {detail['blocked']} (Day {detail['day']}, Hour {detail['hour']:.1f})")

            total_deps = len(all_dependency_arcs)
            response_parts.append(f"\n*Analysis based on {total_deps} total dependency arcs.*")

            return "\n".join(response_parts)

        @tool
        def identify_object_bottlenecks(time_for_query: str = "") -> str:
            """Identify objects that are bottlenecks based on being intermediary nodes."""
            time_param = time_for_query if time_for_query else None

            if not manager.arc_states:
                return "No arc data available for bottleneck analysis."

            day, hour = agent._parse_time_query(time_param)
            start_idx = agent._find_arc_state_index_for_time(day, hour)

            if start_idx is None:
                start_idx = 0

            all_dependency_arcs = []
            for i in range(start_idx, len(manager.arc_states)):
                state = manager.arc_states[i]
                arcs = state.get("arcs", [])
                for arc in arcs:
                    if arc.get("arc_type") == "dependency":
                        all_dependency_arcs.append({
                            **arc,
                            "state_day": state.get("current_day", 0),
                            "state_hour": state.get("current_hour", 0),
                        })

            if not all_dependency_arcs:
                return f"No dependency arcs found from Day {day}, Hour {hour} onwards. No bottlenecks detected."

            bottleneck_counter: Counter[str] = Counter()
            blocking_details: dict[str, list[dict]] = {}

            for arc in all_dependency_arcs:
                blocked_by = arc.get("blocked_by", "")
                blocked_source = arc.get("source", "unknown")
                blocked_target = arc.get("target", "unknown")

                if "->" in blocked_by:
                    parts = blocked_by.split("->")
                    intermediary_obj = parts[-1].strip()
                else:
                    intermediary_obj = blocked_by

                if intermediary_obj:
                    bottleneck_counter[intermediary_obj] += 1
                    if intermediary_obj not in blocking_details:
                        blocking_details[intermediary_obj] = []
                    blocking_details[intermediary_obj].append({
                        "blocked": f"{blocked_source} → {blocked_target}",
                        "blocking_task": blocked_by,
                        "day": arc.get("state_day", 0),
                        "hour": arc.get("state_hour", 0),
                    })

            if not bottleneck_counter:
                return "Could not identify bottleneck objects from dependency arcs."

            top_3 = bottleneck_counter.most_common(3)

            response_parts = [f"**Top 3 Bottleneck Objects** (from Day {day}, Hour {hour} onwards):\n"]

            for rank, (obj, count) in enumerate(top_3, 1):
                response_parts.append(f"{rank}. **{obj}** - involved in {count} blocking dependencies")

                details = blocking_details.get(obj, [])[:3]
                for detail in details:
                    response_parts.append(f"   - Task '{detail['blocking_task']}' blocks: {detail['blocked']} (Day {detail['day']}, Hour {detail['hour']:.1f})")

            total_deps = len(all_dependency_arcs)
            response_parts.append(f"\n*Analysis based on {total_deps} total dependency arcs.*")

            return "\n".join(response_parts)

        return [run_text_query_on_map, run_arc_query_on_map, identify_task_bottlenecks, identify_object_bottlenecks]

    def _build_agent(self):
        """Build the Langchain agent with tools."""
        llm = self._get_llm()
        tools = self._create_tools()
        system_prompt = self._get_system_prompt()

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

        return agent

    def chat(self, message: str, chat_history: list | None = None) -> str:
        """Process a chat message and return the agent's response."""
        try:
            if chat_history is not None:
                messages = chat_history + [{"role": "user", "content": message}]
            else:
                messages = self.chat_manager.get_context_messages()
                if not messages or messages[-1].get("content") != message:
                    messages = messages + [{"role": "user", "content": message}]

            result = self._agent_executor.invoke({
                "messages": messages
            })
            messages_out = result.get("messages", [])
            if messages_out:
                last_msg = messages_out[-1]
                output = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                return output if output else "I couldn't process that request."
            return "I couldn't process that request."
        except Exception as e:
            log.error("Agent error: %s", e)
            return f"Error processing request: {e}"

    def respond(self, message: str) -> str:
        """Process a chat message and update the chat history."""
        user_text = message.strip()
        if not user_text:
            return ""

        self.chat_manager.add_user_message(user_text)
        response = self.chat(user_text)
        self.chat_manager.add_assistant_message(response)

        return response
