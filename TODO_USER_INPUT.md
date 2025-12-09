# TODO: User Input for Missing ATS Fields

## Problem
When ATS parser detects missing critical fields (email, phone, etc.), the agent should pause and ask the user to provide them via a modal/form.

## Current State
- State schema has `needs_user_input` and `user_input_request` fields added ✅
- Supervisor doesn't check for user input request yet ❌
- ATS critic doesn't set the flag when fields are missing ❌
- UI doesn't show modal for user input ❌

## Implementation Steps

### 1. ATS Critic - Detect Missing Fields
In `src/agents/JobSeeker/critics.py`, after ATS parsing:

```python
# After line 139: ats_result = check_ats_compatibility(pdf_path)
missing_fields = ats_result.get("missing_critical_fields", [])
if missing_fields and revision_count == 0:  # Only ask on first iteration
    print(f"  ⚠️ Missing critical fields: {missing_fields}")
    print(f"  🛑 Pausing workflow to request user input...")
    return {
        "critique_inputs": [],
        "needs_user_input": True,
        "user_input_request": {
            "type": "missing_ats_fields",
            "fields": missing_fields,
            "message": f"Please provide the following information: {', '.join(missing_fields)}"
        }
    }
```

### 2. Supervisor - Check for User Input Request
In `src/agents/JobSeeker/supervisor.py`, at the start:

```python
# After line 52
# Check if we're waiting for user input
if state.get("needs_user_input"):
    print("  ⏸️ Workflow paused - waiting for user input")
    return Command(
        goto="__end__",  # Pause the workflow
        update={"next_node": "__end__"}
    )
```

### 3. UI - Show Modal and Resume
In `src/main.py`, after `graph.invoke()`:

```python
# After line 48
final_state = graph.invoke(initial_state)

# Check if user input is needed
if final_state.get("needs_user_input"):
    request = final_state.get("user_input_request", {})
    st.warning(request.get("message", "Additional information needed"))

    # Show input form
    with st.form("user_input_form"):
        user_data = {}
        for field in request.get("fields", []):
            user_data[field] = st.text_input(f"Enter {field}:")

        if st.form_submit_button("Continue"):
            # Update resume text with provided data
            updated_resume = final_state["resume_text"]
            for field, value in user_data.items():
                if value:
                    updated_resume += f"\n{field.title()}: {value}"

            # Resume workflow with updated data
            initial_state["resume_text"] = updated_resume
            initial_state["needs_user_input"] = False
            final_state = graph.invoke(initial_state)
```

### 4. Alternative: Interrupt API (LangGraph)
Use LangGraph's built-in interrupt mechanism:

```python
# In graph.py, when building the graph:
graph = workflow.compile(
    checkpointer=MemorySaver(),  # Required for interrupts
    interrupt_before=["supervisor"]  # Can interrupt before supervisor
)

# In UI:
config = {"configurable": {"thread_id": "1"}}
for chunk in graph.stream(initial_state, config):
    if "__interrupt__" in chunk:
        # Show modal, collect input
        # Resume with: graph.stream(None, config, update=user_input)
```

## Testing
1. Upload CV with missing email/phone
2. Verify modal appears
3. Fill in missing data
4. Verify workflow continues with updated data
5. Check final CV includes the new information

## Notes
- Only ask for user input on first iteration (revision 0)
- Don't interrupt workflow repeatedly
- Validate user input before continuing
- Consider storing user data in session state for persistence
