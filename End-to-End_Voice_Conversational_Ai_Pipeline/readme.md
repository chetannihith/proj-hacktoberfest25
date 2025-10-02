# Hello and welcome to EchoVerse!
(Name of this pipeline is EchoVerse)

Think of EchoVerse as a friendly AI assistant you can have a real conversation with, right from your computer. You don't need to type anything – just speak your mind, ask a question, and it will listen and talk right back to you. It’s a simple, hands-free way to chat with an AI.

So, how does it all work?

It's a simple four-step process:

It Listens: First, the app listens to you through your microphone for a few seconds to catch what you're saying.

It Understands: Next, it takes your voice recording and cleverly figures out the words you spoke, turning them into text.

It Thinks: That text is then sent to a powerful AI brain (Google's Gemini) which comes up with a smart and helpful response.

It Speaks: Finally, EchoVerse turns that text response into natural-sounding speech and plays it back for you to hear.

Ready to give it a try? Here’s how to get it set up.

First, there are a couple of things you'll need on your computer. Make sure you have Python installed, as that's the language EchoVerse is written in. It also needs a tool called FFmpeg to handle the audio properly.

Once that's sorted, you'll want to get the project files. You can do this by cloning the repository from GitHub. After that, you'll need to install all the libraries it depends on, which is as simple as running one command to install everything listed in the requirements file.

The last setup step is to give EchoVerse its "brain." It uses Google's Gemini AI, which requires a special key to use. You can get one for free from Google AI Studio. Once you have your key, just create a new file named .env in the project folder and paste the key in there, naming it GEMINI_API_KEY.

How to start a conversation

Once everything is set up, starting a chat is super easy. Just run the main script from your terminal:

python main.py

The moment you run it, the app will start listening. Speak clearly, and after a few seconds, it will process everything and you'll hear its voice in response! If you find you need more or less time to speak, you can easily change the recording duration right inside the script.