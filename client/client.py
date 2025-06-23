import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available tools"""
        chat = self.model.start_chat(history=[])

        response = await self.session.list_tools()
        # Convert MCP tools to Gemini's FunctionDeclaration format
        available_tools = []
        for tool in response.tools:
            try:
                # Convert inputSchema to Gemini's expected format
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    # Convert JSON schema to Gemini's parameter format
                    if isinstance(tool.inputSchema, dict):
                        # Clean up properties to only include Gemini-supported fields
                        clean_properties = {}
                        if 'properties' in tool.inputSchema:
                            for prop_name, prop_schema in tool.inputSchema['properties'].items():
                                if isinstance(prop_schema, dict):
                                    # Only include fields that Gemini supports
                                    clean_prop = {}
                                    if 'type' in prop_schema:
                                        clean_prop['type'] = prop_schema['type']
                                    if 'description' in prop_schema:
                                        clean_prop['description'] = prop_schema['description']
                                    if 'enum' in prop_schema:
                                        clean_prop['enum'] = prop_schema['enum']
                                    # Add other supported fields as needed
                                    clean_properties[prop_name] = clean_prop
                        
                        parameters = {
                            "type": "object",
                            "properties": clean_properties,
                            "required": tool.inputSchema.get("required", [])
                        }
                
                gemini_tool = {
                    "function_declarations": [{
                        "name": tool.name,
                        "description": tool.description or f"Tool: {tool.name}",
                        "parameters": parameters
                    }]
                }
                available_tools.append(gemini_tool)
            except Exception as e:
                print(f"Warning: Could not convert tool {tool.name}: {e}")
                continue

        # Initial Gemini API call
        try:
            if available_tools:
                response = chat.send_message(
                    query,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1000,
                    ),
                    tools=available_tools
                )
            else:
                print("No tools available")
                response = chat.send_message(
                    query,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1000,
                    )
                )
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"

        # Process response and handle tool calls
        final_text = []

        try:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    final_text.append(part.text)
                elif hasattr(part, 'function_call'):
                    tool_name = part.function_call.name
                    tool_args = part.function_call.args
                    
                    # Execute tool call
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                        # Continue conversation with tool results
                        chat.send_message(f"Tool result: {result.content}")

                        # Get next response from Gemini
                        next_response = chat.send_message(
                            f"Tool {tool_name} returned: {result.content}",
                            generation_config=genai.types.GenerationConfig(
                                max_output_tokens=1000,
                            )
                        )

                        final_text.append(next_response.text)
                    except Exception as e:
                        final_text.append(f"Error executing tool {tool_name}: {str(e)}")
        except Exception as e:
            final_text.append(f"Error processing response: {str(e)}")

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    import os
    asyncio.run(main())