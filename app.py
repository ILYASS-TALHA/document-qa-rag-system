"""Main application entry point for the Document RAG Chat Assistant."""
import asyncio
import time
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Import refactored logic modules
from src.config import DEFAULT_SYSTEM_MESSAGE
import src.ui_components as ui
import src.document_processor as dp
import src.llm_service as llm

# Setup asyncio event loop for streamlit environment compatibility
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def translate_message_handler(message: AIMessage):
    """Handler to translate an AI response into Moroccan Darija and rerun Streamlit."""
    with st.spinner("Traduction en darija..."):
        try:
            chat_model = llm.get_chat_model()
            translation = llm.translate_to_darija(chat_model, message.content)
            
            # Update the message metadata
            new_kwargs = dict(message.additional_kwargs) if message.additional_kwargs else {}
            new_kwargs["translation_darija"] = translation
            message.additional_kwargs = new_kwargs
            st.rerun()
        except Exception as e:
            st.error(f"Erreur de traduction : {e}")

def handle_document_processing_ui(uploaded_files, chat_model):
    """Render document processing trigger button and orchestrate loading state."""
    if st.button("🚀 Process Documents", type="primary", key="process_docs_btn"):
        if not uploaded_files:
            st.error("❌ Please upload at least one document first!")
            return
        if not chat_model:
            st.error("❌ Model not initialized properly. Please check your API key.")
            return

        with st.spinner("Processing documents..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                all_documents = []
                num_files = len(uploaded_files)

                for file_idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"🔄 Saving {uploaded_file.name} ({file_idx+1}/{num_files})...")
                    progress_bar.progress(int((file_idx / num_files) * 50))

                    file_docs = dp.load_document(uploaded_file.name, uploaded_file.getvalue())
                    all_documents.extend(file_docs)

                if not all_documents:
                    raise ValueError(
                        "No text could be extracted from any of the uploaded documents. "
                        "Please upload text-based PDFs or .txt files."
                    )

                status_text.text("✂️ Splitting documents into chunks...")
                progress_bar.progress(60)
                chunks = dp.split_documents(all_documents)

                status_text.text("🧠 Creating embeddings...")
                progress_bar.progress(80)
                vector_store = dp.create_vector_store(chunks)
                retriever = vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": 6}
                )

                status_text.text("🤖 Analyzing document genre and generating questions...")
                progress_bar.progress(95)
                
                # Analyze first chunk to infer genre & suggested questions
                analysis = llm.analyze_document_genre_and_questions(chat_model, chunks[0].page_content)
                st.session_state["document_genre"] = analysis.get("genre", "Document")
                st.session_state["suggested_questions"] = analysis.get("questions", [])

                st.session_state["retriever"] = retriever
                st.session_state["vector_store"] = vector_store
                st.session_state["document_names"] = [f.name for f in uploaded_files]
                st.session_state["document_chunks"] = chunks

                progress_bar.empty()
                status_text.empty()
                st.success("✅ Documents processed! Ready for questions.")
                time.sleep(1.5)
                st.rerun()

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error processing documents: {str(e)}")

def handle_user_input_ui(chat_model):
    """Handle processing of new user questions or suggested queries."""
    prompt = st.chat_input("Ask a question about the document...")
    
    if st.session_state.get("clicked_suggestion"):
        prompt = st.session_state["clicked_suggestion"]
        st.session_state["clicked_suggestion"] = None

    if prompt:
        if not prompt.strip():
            st.warning("Please type a message before sending!")
            return

        st.session_state.messages.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.write(prompt)

        retriever = st.session_state.get("retriever")
        if not retriever:
            with st.chat_message("assistant"):
                error_msg = "❌ Please process a document first to enable question answering."
                st.error(error_msg)
                st.session_state.messages.append(
                    AIMessage(
                        content=error_msg,
                        additional_kwargs={"sources": [], "translation_darija": None}
                    )
                )
            return

        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing document content..."):
                try:
                    # Run RAG QA chain streaming
                    stream_generator, retrieved_docs = llm.run_qa_chain_stream(chat_model, retriever, prompt)
                    
                    message_placeholder = st.empty()
                    full_response = ""

                    for chunk in stream_generator:
                        if chunk and chunk.strip():
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")

                    # Parse and format sources metadata
                    sources = []
                    for doc in retrieved_docs:
                        sources.append({
                            "source_name": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", 0),
                            "content": doc.page_content
                        })

                    if full_response and full_response.strip():
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append(
                            AIMessage(
                                content=full_response,
                                additional_kwargs={"sources": sources, "translation_darija": None}
                            )
                        )
                    else:
                        error_msg = "🚫 No response received. Please try a different model."
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append(
                            AIMessage(
                                content=error_msg,
                                additional_kwargs={"sources": [], "translation_darija": None}
                            )
                        )
                    st.rerun()

                except Exception as e:
                    error_message = str(e).lower()
                    if "not found" in error_message or "invalid" in error_message:
                        error_msg = "❌ This model is not available. Please select a different model."
                    elif "quota" in error_message or "limit" in error_message:
                        error_msg = "📊 API quota exceeded. Please try again later or use a different model."
                    elif "timeout" in error_message:
                        error_msg = "⏱️ Request timed out. Try a different model or try again."
                    else:
                        error_msg = "❌ An error occurred. Try selecting different model or check your api key:("

                    st.error(error_msg)
                    st.session_state.messages.append(
                        AIMessage(
                            content=error_msg,
                            additional_kwargs={"sources": [], "translation_darija": None}
                        )
                    )

def run_analysis_tools(chat_model):
    """Render summary and explanation analysis tools in the UI."""
    vector_store = st.session_state.get("vector_store")
    if vector_store is None:
        return

    st.markdown("---")
    st.markdown("### 🛠️ Outils d'analyse du document (RAG)")
    col_summary, col_explain = st.columns(2)

    with col_summary:
        if st.button("📝 Résumé", use_container_width=True, type="primary", key="summary_tool_btn"):
            with st.chat_message("assistant"):
                with st.spinner("📝 Récupération RAG et génération du résumé..."):
                    try:
                        stream_gen, sources = llm.generate_summary_stream(chat_model, vector_store)
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in stream_gen:
                            if chunk.content:
                                full_response += chunk.content
                                message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                        st.session_state.messages.append(HumanMessage(content="📝 Résumé du document (RAG)"))
                        st.session_state.messages.append(
                            AIMessage(
                                content=full_response,
                                additional_kwargs={"sources": sources, "translation_darija": None}
                            )
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération du résumé : {str(e)}")

    with col_explain:
        if st.button("📖 Explication", use_container_width=True, type="secondary", key="explain_tool_btn"):
            with st.chat_message("assistant"):
                with st.spinner("📖 Récupération RAG et génération de l'explication..."):
                    try:
                        stream_gen, sources = llm.generate_explanation_stream(chat_model, vector_store)
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in stream_gen:
                            if chunk.content:
                                full_response += chunk.content
                                message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                        st.session_state.messages.append(HumanMessage(content="📖 Explication du document (RAG)"))
                        st.session_state.messages.append(
                            AIMessage(
                                content=full_response,
                                additional_kwargs={"sources": sources, "translation_darija": None}
                            )
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération de l'explication : {str(e)}")

def main():
    """Main function executing the Streamlit app logic."""
    init_session_state()
    ui.configure_page()
    ui.apply_custom_styles()
    uploaded_files = ui.render_sidebar()
    
    chat_model = llm.get_chat_model()

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded:")
        for f in uploaded_files:
            st.markdown(f"- **{f.name}** ({f.size} bytes)")

    handle_document_processing_ui(uploaded_files, chat_model)
    run_analysis_tools(chat_model)

    if "document_genre" in st.session_state:
        st.info(f"🏷️ **Genre du document détecté :** {st.session_state['document_genre']}")
        
        suggested_questions = st.session_state.get("suggested_questions", [])
        if len(st.session_state.messages) == 1 and suggested_questions:
            st.write("💡 **Questions suggérées :**")
            cols = st.columns(len(suggested_questions))
            for idx, q in enumerate(suggested_questions):
                if cols[idx].button(q, key=f"sugg_{idx}"):
                    st.session_state["clicked_suggestion"] = q
                    st.rerun()

    ui.display_chat_history(chat_model, translate_message_handler)
    handle_user_input_ui(chat_model)

if __name__ == "__main__":
    main()
