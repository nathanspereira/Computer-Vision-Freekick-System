from pathlib import Path
import tempfile

import streamlit as st

from scripts.run_pipeline import run_video_pipeline

# streamlit wrapper

st.set_page_config(page_title="CVFS Demo", layout="centered")

st.title("CVFS Demo")

uploaded_file = st.file_uploader(
    "Upload a freekick video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_file is not None:
    input_video_bytes = uploaded_file.getvalue()
    st.subheader("Input Video")
    st.video(input_video_bytes)

    if st.button("Process Video", type="primary"):
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("output_video_bytes", None)

        suffix = Path(uploaded_file.name).suffix or ".mp4"

        with st.spinner("Running the computer vision pipeline..."):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir_path = Path(temp_dir)
                    input_path = temp_dir_path / f"input{suffix}"
                    output_video_path = temp_dir_path / "annotated_output.mp4"
                    output_csv_path = temp_dir_path / "tracking_output.csv"

                    input_path.write_bytes(input_video_bytes)

                    pipeline_result = run_video_pipeline(
                        video_path=str(input_path),
                        output_csv_path=str(output_csv_path),
                        output_video_path=str(output_video_path),
                        frames_of_interest=set(),
                        log_all_accepted=True,
                    )

                    st.session_state["pipeline_result"] = pipeline_result

                    if output_video_path.exists() and output_video_path.stat().st_size > 0:
                        st.session_state["output_video_bytes"] = output_video_path.read_bytes()
            except Exception as exc:
                st.session_state["pipeline_result"] = {
                    "success": False,
                    "message": f"Pipeline failed unexpectedly: {exc}",
                    "output_video_path": None,
                    "output_csv_path": None,
                }

    pipeline_result = st.session_state.get("pipeline_result")
    if pipeline_result is not None:
        if pipeline_result["success"]:
            st.success(pipeline_result["message"])
        else:
            st.error(pipeline_result["message"])

        output_video_bytes = st.session_state.get("output_video_bytes")
        if output_video_bytes:
            st.subheader("Processed Annotated Video")
            st.video(output_video_bytes)
            st.download_button(
                "Download Processed Video",
                data=output_video_bytes,
                file_name="freekick_annotated_output.mp4",
                mime="video/mp4",
            )
else:
    st.info("Drop in a local video file to start.")
