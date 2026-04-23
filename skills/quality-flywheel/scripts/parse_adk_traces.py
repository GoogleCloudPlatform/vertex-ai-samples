#!/usr/bin/env python3
"""Parse ADK session traces into Vertex Evaluation SDK dataset format.

Reads serialized ADK session JSON (from Session.model_dump_json() or
DatabaseSessionService exports) and converts to the canonical
EvaluationDataset format for use with client.evals.evaluate().

Usage:
  python parse_adk_traces.py --input session.json --output dataset.json
  python parse_adk_traces.py --input_dir ./sessions/ --output dataset.json
  python parse_adk_traces.py --input session.json  # prints to stdout

Input format: JSON file(s) with ADK Session structure:
  {
    "id": "...", "app_name": "...", "user_id": "...",
    "events": [{"author": "user"|"agent_name", "content": {...}}, ...]
  }

Output format: JSON with EvaluationDataset structure:
  {
    "eval_cases": [{"agent_data": {"agents": {...}, "turns": [...]}}]
  }
"""

import argparse
import json
import os
import sys
from typing import Any


def _is_user_event(event: dict[str, Any]) -> bool:
  """Check if an event is from the user."""
  if event.get("author") == "user":
    return True
  content = event.get("content")
  if isinstance(content, dict) and content.get("role") == "user":
    return True
  if event.get("role") == "user":
    return True
  return False


def _extract_content(event: dict[str, Any]) -> dict[str, Any] | None:
  """Extract genai Content from an ADK event dict."""
  if "content" in event:
    raw = event["content"]
    if isinstance(raw, dict) and "parts" in raw:
      return raw
    if isinstance(raw, str):
      return {
          "role": "user" if _is_user_event(event) else "model",
          "parts": [{"text": raw}],
      }
  if "parts" in event:
    return {"role": event.get("role", "model"), "parts": event["parts"]}
  return None


def _extract_author(event: dict[str, Any], default_agent_id: str) -> str:
  """Extract the author from an event, preserving sub-agent attribution."""
  author = event.get("author")
  if author:
    return author
  content = event.get("content")
  if isinstance(content, dict) and content.get("role") == "user":
    return "user"
  if event.get("role") == "user":
    return "user"
  return default_agent_id


def _extract_agent_configs(
    session: dict[str, Any],
) -> dict[str, dict[str, Any]]:
  """Extract agent configs from session metadata if available."""
  configs = {}
  # Check for agent_config in session metadata
  agent_config = session.get("agent_config") or session.get("agentConfig")
  if agent_config:
    agent_id = agent_config.get("agent_id") or agent_config.get(
        "agentId", "agent"
    )
    configs[agent_id] = {
        "agent_id": agent_id,
        "agent_type": agent_config.get(
            "agent_type", agent_config.get("agentType")
        ),
        "instruction": agent_config.get("instruction"),
        "description": agent_config.get("description"),
    }
    return configs

  # Infer from events — collect unique non-user authors
  events = session.get("events", [])
  authors = set()
  for event in events:
    author = event.get("author", "")
    if author and author != "user":
      authors.add(author)

  if not authors:
    authors.add(session.get("app_name", session.get("appName", "agent")))

  for author in authors:
    configs[author] = {"agent_id": author}

  return configs


def _segment_into_turns(
    events: list[dict[str, Any]], default_agent_id: str
) -> list[dict[str, Any]]:
  """Segment a flat event list into ConversationTurns.

  A new turn starts with each user message (matching AgentData.from_session()
  behavior).
  """
  turns = []
  current_events = []

  for event in events:
    is_user = _is_user_event(event)

    # Start new turn on user message (if we have accumulated events)
    if is_user and current_events:
      turns.append({
          "turn_index": len(turns),
          "turn_id": f"turn_{len(turns)}",
          "events": current_events,
      })
      current_events = []

    content = _extract_content(event)
    if content is None:
      continue

    author = _extract_author(event, default_agent_id)

    agent_event = {"author": author, "content": content}

    # Preserve state_delta if present (from EventActions)
    actions = event.get("actions", {})
    state_delta = actions.get("state_delta") or actions.get("stateDelta")
    if state_delta:
      agent_event["state_delta"] = state_delta

    current_events.append(agent_event)

  # Don't forget the last turn
  if current_events:
    turns.append({
        "turn_index": len(turns),
        "turn_id": f"turn_{len(turns)}",
        "events": current_events,
    })

  return turns


def parse_session(session: dict[str, Any]) -> dict[str, Any]:
  """Convert a single ADK session dict to an EvalCase dict."""
  events = session.get("events", [])
  if not events:
    raise ValueError(f"Session {session.get('id', 'unknown')} has no events.")

  agent_configs = _extract_agent_configs(session)
  default_agent_id = next(iter(agent_configs))

  turns = _segment_into_turns(events, default_agent_id)
  if not turns:
    raise ValueError(
        f"Session {session.get('id', 'unknown')} produced no turns."
    )

  agent_data = {"agents": agent_configs, "turns": turns}
  return {"agent_data": agent_data}


def parse_file(filepath: str) -> list[dict[str, Any]]:
  """Parse a JSON file containing one or more ADK sessions."""
  with open(filepath) as f:
    data = json.load(f)

  # Handle single session or list of sessions
  if isinstance(data, list):
    sessions = data
  elif isinstance(data, dict):
    if "events" in data:
      sessions = [data]
    elif "sessions" in data:
      sessions = data["sessions"]
    else:
      raise ValueError(
          f"Unrecognized format in {filepath}. Expected a session object "
          "with 'events' field, a list of sessions, or an object with "
          "'sessions' field."
      )
  else:
    raise ValueError(f"Unexpected JSON type in {filepath}: {type(data)}")

  eval_cases = []
  for i, session in enumerate(sessions):
    try:
      eval_cases.append(parse_session(session))
    except ValueError as e:
      print(f"WARNING: Skipping session {i}: {e}", file=sys.stderr)

  return eval_cases


def main():
  parser = argparse.ArgumentParser(
      description="Parse ADK session traces into Vertex Eval dataset format."
  )
  parser.add_argument(
      "--input",
      "-i",
      help="Path to a single ADK session JSON file.",
  )
  parser.add_argument(
      "--input_dir",
      "-d",
      help="Path to a directory of ADK session JSON files.",
  )
  parser.add_argument(
      "--output",
      "-o",
      help="Output file path. Prints to stdout if not specified.",
  )
  args = parser.parse_args()

  if not args.input and not args.input_dir:
    parser.error("Specify --input or --input_dir")

  eval_cases = []

  if args.input:
    if not os.path.exists(args.input):
      print(f"ERROR: File not found: {args.input}", file=sys.stderr)
      sys.exit(1)
    eval_cases.extend(parse_file(args.input))

  if args.input_dir:
    if not os.path.isdir(args.input_dir):
      print(f"ERROR: Directory not found: {args.input_dir}", file=sys.stderr)
      sys.exit(1)
    json_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".json")
    )
    if not json_files:
      print(f"ERROR: No .json files in {args.input_dir}", file=sys.stderr)
      sys.exit(1)
    for filename in json_files:
      filepath = os.path.join(args.input_dir, filename)
      eval_cases.extend(parse_file(filepath))

  if not eval_cases:
    print("ERROR: No eval cases produced from input.", file=sys.stderr)
    sys.exit(1)

  dataset = {"eval_cases": eval_cases}
  output_json = json.dumps(dataset, indent=2, default=str)

  if args.output:
    with open(args.output, "w") as f:
      f.write(output_json)
    print(
        f"Wrote {len(eval_cases)} eval case(s) to {args.output}",
        file=sys.stderr,
    )
  else:
    print(output_json)


if __name__ == "__main__":
  main()
