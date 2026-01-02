"""
Streamlit UI component for the Developer Dashboard.
"""
import streamlit as st
import logging
from workflows.pipeline import DevPipeline
from rules.engine import RuleEngine

logger = logging.getLogger(__name__)

def render_dev_dashboard():
    st.header("🛠️ Developer Dashboard")
    st.markdown("Use autonomous agents to accelerate your development workflow.")

    # Initialize pipeline
    if "dev_pipeline" not in st.session_state:
        # Check for API Key first (using existing state or env)
        # For now, we assume environment variables are set or we handle errors gracefully
        try:
            st.session_state.dev_pipeline = DevPipeline()
            st.success("Agents initialized successfully.")
        except Exception as e:
            st.error(f"Failed to initialize agents: {e}")
            return

    tab1, tab2 = st.tabs(["Feature Implementation", "Code Validation"])

    with tab1:
        st.subheader("Implement New Feature")
        feature_desc = st.text_area("Describe the feature:", height=150, 
                                  placeholder="e.g., Create a utility function to calculate mortgage amortization schedule...")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            use_git = st.checkbox("Enable Git Integration", help="Create feature branch and commit changes", value=False)
        
        if st.button("Generate Implementation"):
            if not feature_desc:
                st.warning("Please provide a description.")
            else:
                with st.spinner("Agents are working... (Coding -> Validating -> Testing -> Documenting)"):
                    try:
                        result = st.session_state.dev_pipeline.implement_feature(feature_desc, use_git=use_git)
                        
                        if result["status"] == "success":
                            st.balloons()
                            
                            if use_git:
                                git_steps = [s for s in result.get("steps", []) if s["step"] in ("git_branch", "git_commit")]
                                if git_steps:
                                    st.success("Git Operations Completed:")
                                    for step in git_steps:
                                        if step["step"] == "git_branch":
                                            st.write(f"- 🌿 Created branch: `{step['branch']}`")
                                        elif step["step"] == "git_commit":
                                            st.write(f"- 💾 Committed: `{step['message']}`")

                            st.subheader("Generated Code")
                            st.code(result["final_output"]["code"], language="python")
                            
                            st.subheader("Unit Tests")
                            st.code(result["final_output"]["tests"], language="python")
                            
                            st.subheader("Documentation")
                            st.markdown(result["final_output"]["docs"])
                        else:
                            st.error(f"Pipeline failed: {result.get('error')}")
                            if "steps" in result:
                                for step in result["steps"]:
                                    if step.get("violations"):
                                        st.warning(f"Violations in {step['step']}: {step['violations']}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    with tab2:
        st.subheader("Validate Existing Code")
        code_input = st.text_area("Paste code here:", height=200)
        
        if st.button("Run Rule Engine"):
            if not code_input:
                st.warning("Please paste some code.")
            else:
                engine = RuleEngine()
                violations = engine.validate_code(code_input)
                
                if not violations:
                    st.success("✅ No violations found!")
                else:
                    st.error(f"Found {len(violations)} violations:")
                    for v in violations:
                        st.warning(f"[{v.severity.upper()}] {v.rule_id}: {v.message} (Line {v.line_number})")

