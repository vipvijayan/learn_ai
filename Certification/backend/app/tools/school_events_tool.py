"""
Custom LangChain Tool for searching school events and programs
"""
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
import os
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class SchoolEventsSearchInput(BaseModel):
    """Input for the School Events Search tool."""
    query: str = Field(description="The search query to find relevant school events and programs")


class SchoolEventsSearchTool(BaseTool):
    """Tool for searching school events and programs from email data."""
    
    name: str = "school_events_search"
    description: str = """
    Useful for finding information about school events, programs, and activities.
    Use this tool when parents ask about:
    - Coding programs or technology classes
    - Art classes or creative programs
    - Sports activities or athletic programs
    - Music programs or performances
    - Holiday camps or day camps
    - Registration dates and deadlines
    - Age requirements or grade levels
    - Program schedules and timing
    - Spring break, winter break, or other school holidays
    - District-specific events (Leander, Round Rock, Harmony, etc.)
    
    Input should be a search query describing what kind of event or program information is needed.
    """
    args_schema: Type[BaseModel] = SchoolEventsSearchInput
    
    vector_store: Optional[Chroma] = None
    
    def __init__(self):
        super().__init__()
        self._setup_vector_store()
    
    def _setup_vector_store(self):
        """Initialize the vector store with school events data"""
        try:
            # Load school events data
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            documents = []
            
            # Load all JSON files from data directory
            if os.path.exists(data_dir):
                for filename in os.listdir(data_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(data_dir, filename)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                            # Convert JSON to readable text format
                            content = self._format_event_data(data)
                            metadata = {
                                "source": filename,
                                "type": "school_event"
                            }
                            documents.append(Document(page_content=content, metadata=metadata))
            
            if not documents:
                print("Warning: No JSON files found in data directory")
                return
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="school_events_tool"
            )
            
            print(f"✅ School Events Tool initialized with {len(documents)} documents and {len(chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error initializing School Events Tool: {e}")
            raise e
    
    def _format_event_data(self, data: dict) -> str:
        """Format JSON event data into readable text"""
        lines = []
        
        # Handle different data structures
        if "program_name" in data:
            # It's a program like CodeWizardsHQ or Chess Club
            lines.append(f"Program: {data['program_name']}")
            
            if "organization" in data:
                lines.append(f"Organization: {data['organization']}")
            
            if "category" in data:
                lines.append(f"Category: {data['category']}")
            
            if "age_range" in data:
                lines.append(f"Age Range: {data['age_range']}")
            
            # PRICING INFORMATION - CRITICAL!
            if "pricing" in data:
                pricing = data["pricing"]
                if isinstance(pricing, dict):
                    lines.append("Pricing:")
                    for key, value in pricing.items():
                        formatted_key = key.replace("_", " ").title()
                        lines.append(f"  - {formatted_key}: {value}")
                else:
                    lines.append(f"Pricing: {pricing}")
            
            # Meeting schedules
            if "meeting_schedule" in data:
                schedule = data["meeting_schedule"]
                if isinstance(schedule, dict):
                    lines.append("Meeting Schedule:")
                    for level, time in schedule.items():
                        lines.append(f"  - {level.title()}: {time}")
                else:
                    lines.append(f"Meeting Schedule: {schedule}")
            
            # Program features
            if "program_features" in data and isinstance(data["program_features"], list):
                lines.append("Program Features:")
                for feature in data["program_features"]:
                    lines.append(f"  - {feature}")
            
            # Benefits
            if "benefits" in data and isinstance(data["benefits"], list):
                lines.append("Benefits:")
                for benefit in data["benefits"]:
                    lines.append(f"  - {benefit}")
            
            # Skill levels
            if "skill_levels" in data and isinstance(data["skill_levels"], dict):
                lines.append("Skill Levels:")
                for level, info in data["skill_levels"].items():
                    if isinstance(info, dict):
                        lines.append(f"  - {level.title()}: {info.get('description', '')}")
                        if "topics" in info:
                            lines.append(f"    Topics: {', '.join(info['topics'])}")
            
            if "registration" in data and isinstance(data["registration"], dict):
                reg = data["registration"]
                lines.append("Registration:")
                if "status" in reg:
                    lines.append(f"  - Status: {reg['status']}")
                if "trial_session" in reg:
                    lines.append(f"  - Trial Session: {reg['trial_session']}")
                if "call_to_action" in reg:
                    lines.append(f"  - Call to Action: {reg['call_to_action']}")
                if "link" in reg:
                    lines.append(f"  - Registration Link: {reg['link']}")
            
            # Contact information
            if "contact" in data and isinstance(data["contact"], dict):
                contact = data["contact"]
                lines.append("Contact Information:")
                if "website" in contact:
                    lines.append(f"  - Website: {contact['website']}")
                if "email" in contact:
                    lines.append(f"  - Email: {contact['email']}")
                if "phone" in contact:
                    lines.append(f"  - Phone: {contact['phone']}")
            
            if "details" in data:
                lines.append(f"Additional Details: {json.dumps(data['details'], indent=2)}")
        
        elif "calendar" in data and isinstance(data["calendar"], list):
            # It's a calendar of events
            lines.append("School Calendar Events:")
            for event in data["calendar"]:
                if isinstance(event, dict):
                    date = event.get("date", "Unknown Date")
                    event_type = event.get("type", "Unknown Event")
                    district = event.get("district", "Unknown District")
                    lines.append(f"  - {date}: {event_type} (District: {district})")
        
        elif "event_name" in data:
            # Standard event structure
            lines.append(f"Event: {data['event_name']}")
            
            if "description" in data:
                lines.append(f"Description: {data['description']}")
            
            if "organizer" in data:
                lines.append(f"Organizer: {data['organizer']}")
            
            if "date" in data:
                lines.append(f"Date: {data['date']}")
            if "time" in data:
                lines.append(f"Time: {data['time']}")
            
            if "location" in data:
                lines.append(f"Location: {data['location']}")
            
            if "age_range" in data:
                lines.append(f"Age Range: {data['age_range']}")
            if "grade_levels" in data:
                lines.append(f"Grade Levels: {data['grade_levels']}")
            
            # PRICING INFORMATION - Handle both 'cost' and 'pricing' fields
            if "cost" in data:
                lines.append(f"Cost: {data['cost']}")
            
            if "pricing" in data:
                pricing = data["pricing"]
                if isinstance(pricing, dict):
                    lines.append("Pricing:")
                    for key, value in pricing.items():
                        formatted_key = key.replace("_", " ").title()
                        lines.append(f"  - {formatted_key}: {value}")
                else:
                    lines.append(f"Pricing: {pricing}")
            
            if "registration" in data:
                reg = data["registration"]
                if isinstance(reg, dict):
                    if "deadline" in reg:
                        lines.append(f"Registration Deadline: {reg['deadline']}")
                    if "link" in reg:
                        lines.append(f"Registration Link: {reg['link']}")
                else:
                    lines.append(f"Registration: {reg}")
            
            if "contact" in data:
                contact = data["contact"]
                if isinstance(contact, dict):
                    if "email" in contact:
                        lines.append(f"Contact Email: {contact['email']}")
                    if "phone" in contact:
                        lines.append(f"Contact Phone: {contact['phone']}")
                    if "website" in contact:
                        lines.append(f"Contact Website: {contact['website']}")
                else:
                    lines.append(f"Contact: {contact}")
            
            if "details" in data:
                lines.append(f"Additional Details: {data['details']}")
            
            if "programs" in data and isinstance(data["programs"], list):
                lines.append("Available Programs:")
                for program in data["programs"]:
                    if isinstance(program, dict):
                        prog_name = program.get("name", "Unnamed Program")
                        lines.append(f"  - {prog_name}")
                        if "description" in program:
                            lines.append(f"    Description: {program['description']}")
                    else:
                        lines.append(f"  - {program}")
        else:
            # Fallback: convert entire JSON to string
            lines.append(json.dumps(data, indent=2))
        
        return "\n".join(lines)
    
    def _run(self, query: str) -> str:
        """Search for school events based on the query"""
        logger.info(f"\n🔍 TOOL INVOKED: school_events_search")
        logger.info(f"   Query: {query}")
        
        start_time = datetime.now()
        
        try:
            if not self.vector_store:
                logger.error("   ❌ Vector store not initialized")
                return "Error: School events database is not initialized."
            
            # Perform similarity search
            logger.info(f"   🔎 Performing similarity search (k=3)...")
            results = self.vector_store.similarity_search(query, k=3)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if not results:
                logger.warning(f"   ⚠️  No results found (took {duration:.2f}s)")
                return "No relevant school events or programs found for your query."
            
            logger.info(f"   ✅ Found {len(results)} results (took {duration:.2f}s)")
            for i, doc in enumerate(results, 1):
                logger.info(f"      Result {i}: {doc.metadata.get('source', 'Unknown')} - {len(doc.page_content)} chars")
            
            # Format results
            response_parts = []
            response_parts.append(f"Found {len(results)} relevant school events/programs:\n")
            
            for i, doc in enumerate(results, 1):
                response_parts.append(f"\n📋 Result {i}:")
                response_parts.append(doc.page_content)
                response_parts.append(f"📁 Source: {doc.metadata.get('source', 'Unknown')}")
                if i < len(results):
                    response_parts.append("\n" + "-" * 70)
            
            response = "\n".join(response_parts)
            logger.info(f"   📄 Response length: {len(response)} characters")
            
            return response
            
        except Exception as e:
            logger.error(f"   ❌ Error in tool execution: {str(e)}")
            return f"Error searching school events: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of the run method"""
        return self._run(query)


def create_school_events_tool():
    """Create and return an instance of the School Events Search Tool"""
    return SchoolEventsSearchTool()
