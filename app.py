import streamlit as st
from serpapi import GoogleSearch
import os
from groq import Groq
import json
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Video Search & Transcribe",
    page_icon="🎥",
    layout="wide"
)

# Get API keys from environment
SERP_API = os.environ.get("SERP_API")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Check if API keys are set
if not all([SERP_API, GROQ_API_KEY, GOOGLE_API_KEY]):
    st.error("⚠️ Missing API Keys! Please set environment variables: SERP_API, GROQ_API_KEY, GOOGLE_API_KEY")
    st.stop()

# Initialize clients
@st.cache_resource
def init_clients():
    # Configure Google AI
    gemini_client = genai.configure(api_key=GOOGLE_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    return gemini_client, groq_client

gemini_client, groq_client = init_clients()

# Tool functions
def search_video(video_title: str) -> str:
    """Search for a video and return the top link"""
    params = {
        "engine": "google",
        "q": video_title,
        "api_key": SERP_API
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    
    if not organic_results:
        return "No results found."
    
    return organic_results[0]["link"]

def transcribe_content(video_url: str) -> str:
    """Transcribe and summarize video content using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([
            f"Analyze this YouTube video: {video_url}",
            "Provide a 3-sentence summary of the main content."
        ])
        return response.text
    except Exception as e:
        return f"Error transcribing video: {str(e)}"

# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "VideoSearchTool",
            "description": "Search and get the YouTube link for a video by title",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_title": {
                        "type": "string",
                        "description": "The title or topic of the video to search for"
                    }
                },
                "required": ["video_title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "TranscriptionTool",
            "description": "Transcribe and summarize a YouTube video from its URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_url": {
                        "type": "string",
                        "description": "The full YouTube URL"
                    }
                },
                "required": ["video_url"]
            }
        }
    }
]

available_tools = {
    "VideoSearchTool": search_video,
    "TranscriptionTool": transcribe_content
}

# Streamlit UI
st.title("🎥 Video Search & Transcription AI")
st.markdown("Search for videos and get AI-powered transcriptions and summaries!")

# Main input
user_input = st.text_input(
    "What would you like to do?",
    placeholder="e.g., Search and transcribe a video about Windows installation",
    key="user_query"
)

# Process button
if st.button("🚀 Process", type="primary", use_container_width=True):
    if not user_input:
        st.warning("⚠️ Please enter a query!")
    else:
        # Create containers for output
        progress_container = st.container()
        result_container = st.container()
        
        with progress_container:
            with st.spinner("🤖 AI is thinking..."):
                # Initialize messages
                messages = [
                    {
                        "role": "system",
                        "content": """You are a helpful assistant with access to tools.

                        When asked to search video, use VideoSearchTool to search for the video

                        When asked to transcribe, use TranscriptionTool
                        
                        When asked to search and transcribe a video:
                        1. First use VideoSearchTool to find the video URL
                        2. Then use TranscriptionTool with that URL to get the transcript
                        3. Work step-by-step"""
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
                
                # Initial API call
                chat_completion = groq_client.chat.completions.create(
                    messages=messages,
                    tools=tools,
                    model="openai/gpt-oss-120b",
                    tool_choice="auto"
                )
                
                response = chat_completion.choices[0].message
                
                # Create expander for tool execution logs
                with st.expander("🔧 Tool Execution Log", expanded=True):
                    log_placeholder = st.empty()
                    log_text = ""
                    
                    if response.tool_calls:
                        max_iterations = 10
                        iteration = 0
                        
                        while response.tool_calls and iteration < max_iterations:
                            iteration += 1
                            messages.append(response)
                            
                            log_text += f"\n**Iteration {iteration}:** Model called {len(response.tool_calls)} tool(s)\n\n"
                            
                            for tool_call in response.tool_calls:
                                function_name = tool_call.function.name
                                function_args = json.loads(tool_call.function.arguments)
                                
                                log_text += f"- 🔹 **{function_name}** `{function_args}`\n"
                                log_placeholder.markdown(log_text)
                                
                                # Execute function
                                try:
                                    function_to_call = available_tools[function_name]
                                    function_response = function_to_call(**function_args)
                                    
                                    # Truncate long responses in log
                                    display_response = function_response[:100] + "..." if len(function_response) > 100 else function_response
                                    log_text += f"  - ✅ Response: `{display_response}`\n\n"
                                    
                                except Exception as e:
                                    function_response = f"Error: {str(e)}"
                                    log_text += f"  - ❌ Error: `{str(e)}`\n\n"
                                
                                log_placeholder.markdown(log_text)
                                
                                # Add tool result to messages
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": function_name,
                                    "content": str(function_response)
                                })
                            
                            # Get next response
                            chat_completion = groq_client.chat.completions.create(
                                model="openai/gpt-oss-120b",
                                messages=messages,
                                tools=tools,
                                tool_choice="auto"
                            )
                            response = chat_completion.choices[0].message
                        
                        if iteration >= max_iterations:
                            log_text += "\n⚠️ *Reached maximum iterations*\n"
                            log_placeholder.markdown(log_text)
        
        # Display final result
        with result_container:
            st.divider()
            st.subheader("💬 AI Response")
            
            with st.chat_message("assistant"):
                st.markdown(response.content)


#     messages.append(response)

#     for tool_call in tools_calls:
#         function_name = tool_call.function.name
#         function_to_call = available_tools[function_name]

#         function_args = json.loads(tool_call.function.arguments)
#         function_response = function_to_call(**function_args)

#         messages.append(
#             {
#                 "tool_call_id": tool_call.id,
#                 "role": "tool",
#                 "name": function_name,
#                 "content": function_response
#             }
#         )

#     final_response = client.chat.completions.create(
#         messages=messages,
#         model="openai/gpt-oss-120b",
#         tools=tools,
#         tool_choice="auto"
#     )

#     print(final_response.choices[0].message.content)

# else:
#     print(response.content)
