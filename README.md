
<h1 align="center">
  <br>
    <br>
    Challenge Helbling
    <br>
</h1>

<h4 align="center">A voice enhancement and memory management relay for a conversational agent.</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#related">Related</a> •
  <a href="#support">Support</a>
</p>

## Key Features

* API interaction with Helbling VoiceBot Hemy
* STT using ..., strictly focused on the main user
* Memory handling to remember and inject past conversational details

## Challenge Topics
- Quality (50%)
  - The quality of the solution is key. The main voice does not have to be completely isolated, but it should work well in the speech to text process. We identify different tasks:
    - Isolation of the main voice over background noise and voices. (15%)
    - Labelling of different voices. (10%)
    - Extraction and storing of conversational information “memories” per voice. (10%)
    - Injection of memories into conversation for emotional and informational enhancement of conversation. (10%)
    - Safety & Privacy of stored information. (5%)
- Presentation (20%)
- Performance (20%)
  - Ideally, input preprocessing should work in real-time on sliced audio samples in the cloud or on a tablet. It should not take more than 0.5s per sample on consumer grade hardware.
- Business (10%)
  - What are the privacy and safety requirements for such a solution?
  - What is the environmental impact of the software?

## How To Use

Example code is provided as a starting point. Areas where additional code is needed are marked with TODOs.

### Relay Setup

You can set up the relay locally using the following lines. A Docker container is preferable but not yet available.
You can also set up your own relay from scratch using the api description: [apispec.html](./docs/apispec.html) or run `src/relay.py` and open [apidocs](http://localhost:5000/apidocs/).

#### Local build

Requirements: Python3.12 with pip and pipx installed, an IDE, for example Visual Studio Code.

```bash
# Install poetry (python package manager)
$ pipx install poetry

# Install dependencies
$ poetry install

# Launch virtual environment
$ poetry shell

# Run the app
$ python ./src/relay.py
```

#### Docker build

Docker setup is not available at this time.

### Webapp Setup

Start the webapp [voiceoasis.azurewebsites.net](https://voiceoasis.azurewebsites.net/), allow microphone access, start using the recording button at bottom right and see the python logs for incoming requests.

Note: The microphone button may not function correctly if pressed too quickly in succession. This issue does not require further investigation.

Optional webapp url parameters:
* relayURL=ip:port: ip:port of the relay endpoint, only http supported, default is localhost:5000, 
* audioChunkLengthMS=number: number being a chunk length in milliseconds, default is 500
* continuousListening=1: always listen to the microphone input.
* micDeviceId=n: n being a number of your microphone devices, choose which non-default microphone device to use.
* micDeviceLabel=label: label being a microphone device name
* lang=language: language being one of en-US, de-DE, fr-FR, it-IT, es-ES, ja-JP, default language to show at start
* micButtonStopOnRelease=1: press and hold the microphone button during input instead of pressing it twice. This feature may not always work as expected.

## Related

[Deep Dive Slides](./DeepDiveSlides_Helbling.pdf)

* Microsoft speech key: https://starthack.eu/#/case-details?id=21, Case Description
* Azure OpenAI endpoint: See https://starthack.eu/#/case-details?id=21, Case Description
* Open source voice and transcription data: See for example https://github.com/google-deepmind/librispeech-long


### Generate swagger html docs
Run `src/relay.py` and open [localhost:5000/apispec_1.json](http://localhost:5000/apispec_1.json). Download the content to `docs/apispec.json`.

Use following commands to generate the self-contained html (ensure npm is installed):

```
npm install -g redoc-cli
redoc-cli bundle docs/apispec.json -o docs/apispec.html
```

## Support

Visit us at our Helbling Booth in the Hack area or drop us a message in discord.
