We both are planning to work on an assignment

The plan is as follows

1. Write 1-2 paragraphs on why this is a problem for your specific user:

The problem we are trying to solve is a problem faced by parents.
Parents gets a lot of email from schools about different events and programs.
Its hard to remember all these.

2. Propose a Solution

I am creating a tool where the user can ask AI to find events and programs based on user's choice by looking at all the data from schools.

3. Describe the tools you plan to use in each part of your stack.  Write one sentence on why you made each tooling choice. 

**Tavily**: Tavily used to search the events and programs from schools online


4. Where will you use an agent or agents?  What will you use “agentic reasoning” for in your app? 

5. Describe all of your data sources and external APIs, and describe what you’ll use them for.

- All data is currently saved in the data folder.
- Tavily tool is also used to search online.

6. Describe the default chunking strategy that you will use.  Why did you make this decision?

Chunk size is now 1000 with overlap of 200 because currently I have smaller data set to query from. 

I use RecursiveCharacterTextSplitter with the following configuration:

Chunk size: 1000 characters
Chunk overlap: 200 characters
Splitter type: RecursiveCharacterTextSplitter

Why We Made This Decision:
Balanced Chunk Size:

1000 characters provides enough context for the embedding model to understand each piece while staying well within token limits
Not too large (keeps embeddings focused) and not too small (preserves context)


7. Assess your pipeline using the RAGAS framework including key metrics faithfulness, response relevance, context precision, and context recall. Provide a table of your output results.

8. What conclusions can you draw about the performance and effectiveness of your pipeline with this information?

    - Shown in the UI

9. Swap out base retriever with advanced retrieval methods.

    - Updated Code with Naive.

10. How does the performance compare to your original RAG application? Test the new retrieval pipeline using the RAGAS frameworks to quantify any improvements. Provide results in a table.

