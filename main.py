

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Rachel Beard",
        page_icon="👋",
    )

    st.write("# Cluster detection Analysis")

    #st.sidebar.success("Select a task")

    st.markdown(
        """
        This web app demonstrates various interactive tasks for visual explorations to aid decision making in Public Health 
        """
    )

 
 


if __name__ == "__main__":
    run()
