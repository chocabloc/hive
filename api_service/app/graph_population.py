import asyncio
from typing import TypedDict, List, Dict, Optional, Any
from langgraph.graph import StateGraph, END, START
from .tools.tools_population import (
    parse_transcript_and_populate_lines,
    group_lines_for_summarization,
    summarize_line_groups,
    extract_info_json,
    save_summaries_and_links,
    save_and_link_facts,
    extract_graph_from_transcript,
    populate_kg,
    ParsedLine 
)

# --- 1. Define the State ---
class PopulationState(TypedDict):
    transcript: str 
    parsed_lines: List[ParsedLine] 
    
    # Branch A output
    summaries_with_links: List[Dict[str, Any]]
    
    # Branch B output
    structured_data: Optional[Dict]
    
    # Final status
    summary_db_status: Optional[str]
    facts_db_status: Optional[str] # Renamed for clarity

# --- 2. Define the Nodes ---

# Node 1a (Branch A): Parse the transcript and populate the 'lines' table
def populate_lines_node(state: PopulationState):
    print("--- [Graph] Node: 1. parse_transcript_and_populate_lines ---")
    transcript = state['transcript']
    parsed_lines = parse_transcript_and_populate_lines(transcript)
    if not parsed_lines:
        raise ValueError("Failed to parse transcript. Check transcript format.")
    return {"parsed_lines": parsed_lines}

# Node 1b (Branch B): Build knowledge graph links from the transcript
def knowledge_graph_node(state: PopulationState):
    print("--- [Graph] Node: knowledge_graph_links ---")
    transcript = state['transcript']
    populate_kg(transcript)
    return {}

# Node 2a (Branch A): Group lines, summarize them
async def group_and_summarize_node(state: PopulationState):
    print("--- [Graph] Node: 2a. group_and_summarize ---")
    parsed_lines = state['parsed_lines']
    line_groups = group_lines_for_summarization(parsed_lines)
    summaries_with_links = await summarize_line_groups(line_groups)
    return {"summaries_with_links": summaries_with_links}

# Node 2b (Branch B): Extract events/actions
async def extract_info_node(state: PopulationState):
    print("--- [Graph] Node: 2b. extract_info ---")
    transcript = state['transcript']
    structured_data = await extract_info_json(transcript)
    return {"structured_data": structured_data}



# Node 3a (Branch A): Save the summaries and their links
def save_summaries_node(state: PopulationState):
    print("--- [Graph] Node: 3a. save_summaries ---")
    summaries = state['summaries_with_links']
    status = save_summaries_and_links(summaries)
    return {"summary_db_status": status}

# Node 3b (Branch B): Save events/actions AND link them to lines
def save_and_link_facts_node(state: PopulationState):
    print("--- [Graph] Node: 3b. save_and_link_facts ---")
    structured_data = state['structured_data']
    parsed_lines = state['parsed_lines'] # <-- Gets lines from Step 1
    
    status = save_and_link_facts(structured_data, parsed_lines)
    return {"facts_db_status": status}


# --- 3. Build the Graph ---
workflow = StateGraph(PopulationState)

# Add all nodes
workflow.add_node("populate_lines", populate_lines_node)
workflow.add_node("group_and_summarize", group_and_summarize_node)
workflow.add_node("extract_info", extract_info_node)
workflow.add_node("save_summaries", save_summaries_node)
workflow.add_node("save_and_link_facts", save_and_link_facts_node) # <-- ADDED
workflow.add_node("knowledge_graph", knowledge_graph_node)

# --- 4. Define the Workflow ---

# 1. Set the entry point as START node

# 2. First, populate lines from the transcript, and build knowledge graph links parallelly
workflow.add_edge(START, "populate_lines")
workflow.add_edge(START, "knowledge_graph")  # <-- ADDED

# 3. After lines are populated, run summarization and extraction in parallel
workflow.add_edge("populate_lines", "group_and_summarize")
workflow.add_edge("populate_lines", "extract_info")

# 4. Branch A: After summarization, save the summaries
workflow.add_edge("group_and_summarize", "save_summaries")

# 4. Branch B: After extraction, save AND LINK the facts
workflow.add_edge("extract_info", "save_and_link_facts") # <-- UPDATED


# 5. End the graph when both save operations are done
workflow.add_node("join_for_end", lambda x: {})
workflow.add_edge("knowledge_graph", "join_for_end")
workflow.add_edge("join_for_end", END)

population_app = workflow.compile()