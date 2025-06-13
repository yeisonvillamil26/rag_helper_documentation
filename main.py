from chains.retrieval import run_llm
import markdown2
import streamlit as st
import os
from tools.contants import LOCAL_DOCS_PATH

# Configuration
st.header("Documentation Chatbot")


def find_md_file_by_title(title: str) -> str:
    """Looks through our documentation folder to find the Markdown file that best matches
    the given title. This is used to provide context to the LLM and print the document.

    Args:
        title: The document title from OpenSearch (without file extension)
    """
    for root, _, files in os.walk(LOCAL_DOCS_PATH):
        for file in files:
            if file.endswith(".md"):
                # Normalization
                filename_without_ext = (
                    os.path.splitext(file)[0]
                    .lower()
                    .replace("_", "-")
                    .replace(" ", "-")
                )
                normalized_title = title.lower().replace("_", "-").replace(" ", "-")

                if normalized_title in filename_without_ext:
                    return os.path.join(root, file)
    return None


def markdown_to_html(md_filepath: str) -> str:
    """Takes a Markdown file and transforms it into clean HTML with specific formatting,
    making documentation readable and downloadable.

    Args:
        md_filepath: Path to the Markdown file to convert

    Returns:
        A string containing the styled HTML content.

    NOTE: This function is based on the original folder of documentation stored locally
    """
    try:
        with open(md_filepath, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert to HTML
        html_content = markdown2.markdown(md_content)
        styled_html = f"""
        <style>
            .markdown-container {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                padding: 15px;
                background: #f9f9f9;
                border-radius: 5px;
            }}
            .markdown-container pre {{
                background: #f0f0f0;
                padding: 10px;
                border-radius: 3px;
                overflow-x: auto;
            }}
            .markdown-container code {{
                font-family: Consolas, monospace;
            }}
        </style>
        <div class="markdown-container">
            {html_content}
        </div>
        """
        return styled_html
    except Exception as e:
        return f"<div style='color:red'>Error loading file: {str(e)}</div>"


def display_document(title: str, idx: int):
    """Takes a Markdown file and transforms it into clean HTML with pleasant formatting,
    making documentation easier to read right in the chat interface.

    Args:
        title: The document title from OpenSearch
        idx: A unique index used to generate unique keys for Streamlit elements
    """
    md_filepath = find_md_file_by_title(title)

    if md_filepath:
        with st.expander(f"📄 {os.path.basename(md_filepath)}", expanded=False):
            html_content = markdown_to_html(md_filepath)
            st.components.v1.html(html_content, height=400, scrolling=True)

            # Download button with unique key
            with open(md_filepath, "rb") as f:
                st.download_button(
                    label="Download document",
                    data=f,
                    file_name=os.path.basename(md_filepath),
                    mime="text/markdown",
                    key=f"dl_{title}_{idx}",
                )
    else:
        st.error(f"No Markdown file found for: {title}")


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.update(
        {
            "chat_answers_history": [],
            "user_prompt_history": [],
            "chat_history": [],
            "sources_history": [],
        }
    )

prompt = st.text_input("Question", placeholder="Ask your question here...")

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        # Extract document titles from sources
        source_titles = set(
            doc.metadata.get("title", os.path.basename(doc.metadata.get("source", "")))
            for doc in generated_response["source_documents"]
        )

        # Update chat history
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(generated_response["result"])
        st.session_state["sources_history"].append(source_titles)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

# Display chat history
if st.session_state["chat_answers_history"]:
    for i in reversed(range(len(st.session_state["user_prompt_history"]))):
        user_query = st.session_state["user_prompt_history"][i]
        generated_response = st.session_state["chat_answers_history"][i]
        source_titles = st.session_state["sources_history"][i]

        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            st.markdown(generated_response)

            # Show related documentation
            if source_titles:
                st.markdown(
                    """<h4 style='margin-top:20px; margin-bottom:10px'>
                               Related Documentation</h4>""",
                    unsafe_allow_html=True,
                )
                for j, title in enumerate(sorted(source_titles)):
                    display_document(title, i * 100 + j)

        st.divider()
