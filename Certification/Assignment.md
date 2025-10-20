We both are planning to work on an assignment

The plan is as follows

1. Write 1-2 paragraphs on why this is a problem for your specific user:

    The problem we are trying to solve is a problem faced by parents.
    Parents gets a lot of email from schools about different events and programs.
    Its hard to remember all these.

2. Propose a Solution

    I am creating a tool where the user can ask AI to find events and programs based on user's choice by looking at all the data from schools.

3. Describe the tools you plan to use in each part of your stack.  Write one sentence on why you made each tooling choice. 

    - Tavily used to search the events and programs from schools online


4. Where will you use an agent or agents?  What will you use “agentic reasoning” for in your app?

I am using a ReAct (Reasoning and Acting) agent with the Tavily search tool integrated into the RAG pipeline. The agent is created using LangChain's create_react_agent function and executed via AgentExecutor.

The agent autonomously decides whether to use Tavily search for real-time online information or rely on the local vector store, supplementing static data with current information.

First, search the local knowledge base for existing events
If information is incomplete or outdated, use Tavily to search for current events online
Synthesize information from both sources to provide comprehensive answers

5. Describe all of your data sources and external APIs, and describe what you’ll use them for.

    - All data is currently saved in the data folder.
    - Tavily tool is also used to search online.

6. Describe the default chunking strategy that you will use.  Why did you make this decision?

    I use RecursiveCharacterTextSplitter with the following configuration:

    - **Chunk size**: 1000 characters
    - **Chunk overlap**: 200 characters
    - **Splitter type**: RecursiveCharacterTextSplitter
    - **Separators**: ["\n\n", "\n", ",", " ", ""]

    **Why We Made This Decision:**
    
    1. **Improved Retrieval Accuracy**: 
       - Without chunking, each JSON file was stored as a single large document (65 documents total)
       - This caused poor semantic matching for specific queries
       - For example, "Karate Classes for Youth by Dragon Martial Arts Studio" couldn't be found in the top-10 results
       - After implementing chunking, the 65 documents were split into 145 smaller chunks, significantly improving retrieval
    
    2. **Balanced Chunk Size**:
       - 1000 characters provides enough context for the embedding model to understand each piece while staying well within token limits
       - Not too large (keeps embeddings focused on specific content) and not too small (preserves contextual relationships)
    
    3. **Smart Overlap Strategy**:
       - 200-character overlap ensures that related information spanning chunk boundaries isn't lost
       - Helps maintain coherence across chunk boundaries, especially for multi-field JSON structures
    
    4. **JSON-Aware Separators**:
       - Custom separator sequence ["\n\n", "\n", ",", " ", ""] respects JSON structure
       - Prioritizes splitting at natural boundaries (double newlines, single newlines, commas) before breaking mid-word
       - This preserves the integrity of JSON key-value pairs and nested structures

    **Results**: The chunking strategy improved retrieval from 65 large documents to 145 focused chunks, enabling successful retrieval of specific event information that was previously unreachable.


7. Assess your pipeline using the RAGAS framework including key metrics faithfulness, response relevance, context precision, and context recall. Provide a table of your output results.

    - Shown in the UI

8. What conclusions can you draw about the performance and effectiveness of your pipeline with this information?

    - Shown in the UI

9. Swap out base retriever with advanced retrieval methods.

    - Updated Code with Naive.

10. How does the performance compare to your original RAG application? Test the new retrieval pipeline using the RAGAS frameworks to quantify any improvements. Provide results in a table.

    - Shown in UI