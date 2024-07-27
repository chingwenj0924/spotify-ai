import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import random
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Optional, Type
from langchain.llms import Bedrock
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from dotenv import load_dotenv
import os

import boto3

load_dotenv()

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_access_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

# instantitate spotipy client
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id,
                                                                             client_secret=client_secret))

# LangChain Agents make it easy to work with external APIs such as Spotipy. 
# We use a ReAct Agent for our reasoning workflow. LangChain comes with many Tools that are built-in, 
# at the moment of this example there is no built-in tool for the Spotify API, 
# so we [ build our own custom tool ] that we then give our Agent access to. 
# This will enable the Agent to [ take the appropriate actions based off of the input.]

class MusicInput(BaseModel):
    artists: list = Field(description="A list of artists that they'd like to see music from")
    tracks: int = Field(description="The number of tracks/songs they want returned.")
    

# we provide a description of when to use this Tool, 
# this allows for the LLM to use natural language understanding to infer what Tool to use when.
# We also provide a schema of the inputs that the tool should expect. 

class SpotifyTool(BaseTool):
    name = "Spotify Music Recommender"
    description = "Use this tool when asked music recommendations."
    args_schema: Type[BaseModel] = MusicInput  # Pydantic model class to validate and parse the tool’s input arguments.
    
    # utils
    # These methods essentially take the artists that you have retrieved 
    # and return the top tracks of those artists. 
    # Note that currently for the Spotipy API only the top 10 tracks can be retrieved.


    @staticmethod
    def retrieve_id(artist_name: str) -> str:
        results = sp.search(q='artist:' + artist_name, type='artist')
        if len(results) > 0:
            artist_id = results['artists']['items'][0]['id']
        else:
            raise ValueError(f"No artists found with this name: {artist_name}")
        return artist_id

    @staticmethod
    def retrieve_tracks(artist_id: str, num_tracks: int) -> list:
        if num_tracks > 10:
            raise ValueError("Can only provide up to 10 tracks per artist")
        tracks = []
        top_tracks = sp.artist_top_tracks(artist_id)
        for track in top_tracks['tracks'][:num_tracks]:
            tracks.append(track['name'])
        return tracks

    @staticmethod
    def all_top_tracks(artist_array: list) -> list:
        complete_track_arr = []
        for artist in artist_array:
            artist_id = SpotifyTool.retrieve_id(artist)
            all_tracks = {artist: SpotifyTool.retrieve_tracks(artist_id, 10)}
            complete_track_arr.append(all_tracks)
        return complete_track_arr

    # main execution
    # We then define a main execution function where we take all the top tracks of the requested artists 
    # and parse it for the amount of tracks we have requested in our prompt:
    def _run(self, artists: list, tracks: int) -> list:
        num_artists = len(artists)
        max_tracks = num_artists * 10
        print("---------------")
        print(artists)
        print(type(artists))
        print("---------------")
        all_tracks_map = SpotifyTool.all_top_tracks(artists) # map for artists with top 10 tracks
        all_tracks = [track for artist_map in all_tracks_map for artist, tracks in artist_map.items() for track in tracks] #complete list of tracks

        if tracks > max_tracks:
            raise ValueError(f"Only 10 tracks per artist, max tracks for this many artists is: {max_tracks}")
        final_tracks = random.sample(all_tracks, tracks)
        return final_tracks

    def _arun(self):
        raise NotImplementedError("Spotify Music Recommender does not support ")
    


model_id = "anthropic.claude-v2:1"
model_params = {"max_tokens_to_sample": 500,
                "top_k": 100,
                "top_p": .95,
                "temperature": .5}

bedrock_runtime = boto3.client('bedrock-runtime', 'us-east-1', 
                        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
                        aws_access_key_id= aws_access_key,
                        aws_secret_access_key= aws_access_secret,
                        )

llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs=model_params
)

# # sample Bedrock Inference
# llm("What is the capitol of the United States?")


tools = [SpotifyTool()]
agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose = True)

print(agent.run("""I like the following artists: [Arijit Singh, Future, 
The Weekend], can I get 12 song recommendations with them in it."""))
