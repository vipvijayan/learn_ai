<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading"> 👋 Welcome to the AI Engineer Challenge</h1>

## 🤖 Your First Vibe Coding LLM Application

> If you need an introduction to `git`, or information on how to set up API keys for the tools we'll be using in this repository - check out our [Interactive Dev Environment for LLM Development](https://github.com/AI-Maker-Space/Interactive-Dev-Environment-for-AI-Engineers) which has everything you'd need to get started in this repository!

In this repository, we'll walk you through the steps to create a LLM (Large Language Model) powered application with a vibe-coded frontend!

Are you ready? Let's get started!

<details>
  <summary>🖥️ Accessing "gpt-4.1-mini" (ChatGPT) like a developer</summary>

1. Head to [this notebook](https://colab.research.google.com/drive/1sT7rzY_Lb1_wS0ELI1JJfff0NUEcSD72?usp=sharing) and follow along with the instructions!

2. Complete the notebook and try out your own system/assistant messages!

That's it! Head to the next step and start building your application!

</details>

<details>
  <summary>🏗️ Forking & Cloning This Repository</summary>

1. Fork [this](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge) repo!

   ![image](https://i.imgur.com/bhjySNh.png)

1. Clone your newly created repo.

   ```bash
   git clone git@github.com:<YOUR GITHUB USERNAME>/The-AI-Engineer-Challenge.git
   ```

1. Open the freshly cloned repository inside Cursor!

   ```bash
   cd The-AI-Engineering-Challenge
   cursor .
   ```

1. Check out the existing backend code found in `/api/app.py`

</details>

<details>
  <summary> Setting Up for Vibe Coding Success </summary>

While it is a bit counter-intuitive to set things up before jumping into vibe-coding - it's important to remember that there exists a gradient betweeen AI-Assisted Development and Vibe-Coding. We're only reaching _slightly_ into AI-Assisted Development for this challenge, but it's worth it!

1. Check out the rules in `.cursor/rules/` and add theme-ing information like colour schemes in `frontend-rule.mdc`! You can be as expressive as you'd like in these rules!
2. We're going to index some docs to make our application more likely to succeed. To do this - we're going to start with `CTRL+SHIFT+P` (or `CMD+SHIFT+P` on Mac) and we're going to type "custom doc" into the search bar.

   ![image](https://i.imgur.com/ILx3hZu.png)

3. We're then going to copy and paste `https://nextjs.org/docs` into the prompt.

   ![image](https://i.imgur.com/psBjpQd.png)

4. We're then going to use the default configs to add these docs to our available and indexed documents.

   ![image](https://i.imgur.com/LULLeaF.png)

5. After that - you will do the same with Vercel's documentation. After which you should see:

   ![image](https://i.imgur.com/hjyXhhC.png)

</details>

<details>
  <summary>😎 Vibe Coding a Front End</summary>

1. Use `Command-L` or `CTRL-L` to open the Cursor chat console.

2. Set the chat settings to the following:

   ![image](https://i.imgur.com/LSgRSgF.png)

3. Ask Cursor to create a frontend for your application. Iterate as much as you like!

4. Run the frontend using the instructions Cursor provided.

> NOTE: If you run into any errors, copy and paste them back into the Cursor chat window - and ask Cursor to fix them!

</details>

<details>
  <summary>🚀 Deploying Your First LLM-powered Application with Vercel</summary>

1. Ensure you have signed into [Vercel](https://vercel.com/) with your GitHub account.

2. Ensure you have `npm` (this may have been installed in the previous vibe-coding step!) - if you need help with that, ask Cursor!

3. Run the command:

   ```bash
   npm install -g vercel
   ```

4. Run the command:

   ```bash
   vercel
   ```

5. Follow the in-terminal instructions. (Below is an example of what you will see!)

   ![image](https://i.imgur.com/D1iKGCq.png)

6. Once the build is completed - head to the provided link and try out your app!

> NOTE: Remember, if you run into any errors - ask Cursor to help you fix them!

</details>

### 🎉 Congratulations!

You just deployed your first LLM-powered application! 🚀🚀🚀 Get on linkedin and post your results and experience! Make sure to tag us at @AIMakerspace!

Here's a template to get your post started!

```
🚀🎉 Exciting News! 🎉🚀

🏗️ Today, I'm thrilled to announce that I've successfully built and shipped my first-ever LLM using the powerful combination of , and the OpenAI API! 🖥️

Check it out 👇
[LINK TO APP]

A big shoutout to the @AI Makerspace for all making this possible. Couldn't have done it without the incredible community there. 🤗🙏

Looking forward to building with the community! 🙌✨ Here's to many more creations ahead! 🥂🎉

Who else is diving into the world of AI? Let's connect! 🌐💡

#FirstLLMApp
```
# 🧑‍💻 What is [AI Engineering](https://maven.com/aimakerspace/ai-eng-bootcamp)?

Learn more about [The AI Engineering Bootcamp!](https://aimakerspace.io/the-ai-engineering-bootcamp/)

AI Engineering refers to the industry-relevant skills that data science and engineering teams need to successfully **build, deploy, operate, and improve Large Language Model (LLM) applications in production environments**.  

In practice, this requires understanding both prototyping and production deployments.

During the *prototyping* phase, Prompt Engineering, Retrieval Augmented Generation (RAG), Agents, and Fine-Tuning are all necessary tools to be able to understand and leverage. Prototyping includes:
1. Building RAG Applications
2. Building with Agent and Multi-Agent Frameworks
3. Fine-Tuning LLMs & Embedding Models
4. Deploying LLM Prototype Applications to Users

When *productionizing* LLM application prototypes, there are many important aspects ensuring helpful, harmless, honest, reliable, and scalable solutions for your customers or stakeholders. Productionizing includes:
1. Evaluating RAG and Agent Applications
2. Improving Search and Retrieval Pipelines for Production
3. Monitoring Production KPIs for LLM Applications
4. Setting up Inference Servers for LLMs and Embedding Models
5. Building LLM Applications with Scalable, Production-Grade Components

This bootcamp builds on our two previous courses, [LLM Engineering](https://maven.com/aimakerspace/llm-engineering) and [LLM Operations](https://maven.com/aimakerspace/llmops) 👇

- Large Language Model Engineering (LLM Engineering) refers to the emerging best-practices and tools for pretraining, post-training, and optimizing LLMs prior to production deployment.  Pre- and post-training techniques include unsupervised pretraining, supervised fine-tuning, alignment, model merging, distillation, quantization. and others.
    
- Large Language Model Ops (LLM Ops, or LLMOps (as from [WandB](https://docs.wandb.ai/guides/prompts) and [a16z](https://a16z.com/emerging-architectures-for-llm-applications/))) refers to the emerging best-practices, tooling, and improvement processes used to manage production LLM applications throughout the AI product lifecycle.  LLM Ops is a subset of Machine Learning Operations (MLOps) that focuses on LLM-specific infrastructure and ops capabilities required to build, deploy, monitor, and scale complex LLM applications in production environments.  _This term is being used much less in industry these days._

# 🏆 **Grading and Certification**

To become **AI-Makerspace Certified**, which will open you up to additional opportunities for full and part-time work within our community and network, you must:

1. Complete all project assignments.
2. Complete a project and present during Demo Day.
3. Receive at least an 85% total grade in the course.

If you do not complete all assignments, participate in Demo Day, or maintain a high-quality standard of work, you may still be eligible for a *certificate of completion* if you miss no more than 2 live sessions.

# 📚 About

This GitHub repository is your gateway to mastering the art of AI Engineering.  ***All assignments for the course will be released here for your building, shipping, and sharing adventures!***

# 🙏 Contributions

We believe in the power of collaboration. Contributions, ideas, and feedback are highly encouraged! Let's build the ultimate resource for AI Engineering together.

Please to reach out with any questions or suggestions. 

Happy coding! 🚀🚀🚀

