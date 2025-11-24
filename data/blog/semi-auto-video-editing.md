---
title: Semi-Automated Video Editing with Remotion, Whisper, and Gemini AI
date: '2024-06-30'
tags: ['video-editing', 'remotion', 'ai', 'automation', 'gemini', 'whisper', 'workflow']
draft: false
summary: "A deep dive into building a semi-automated video editing pipeline with AI for script, audio, and shot planning."
---
# Semi-Automated Video Editing

# Building a Semi-Automated Video Creation System with AI

*How I combined Remotion, Whisper, and Gemini AI to create an intelligent video editing pipeline*

![Semi-automated video editing system overview showing the workflow from script to final video](placeholder-system-overview.png)

## The Problem

Creating engaging video content is time-consuming and requires significant technical expertise. Traditional video editing workflows involve multiple tools, manual synchronization, and countless hours of fine-tuning. I wanted to build a system that could automate most of the heavy lifting while still allowing for human creativity and quality control.

## The Solution

I built a semi-automated video creation system using **Remotion**, **Whisper**, and **Gemini AI**. While not completely autonomous, it dramatically reduces the manual work required for video production while maintaining quality through human oversight.

<Mermaid chart={`
flowchart TD
    A[Script Generation] --> B[Audio Processing]
    B --> C[Video Generation]
    A --> D[AI Voice Synthesis]
    B --> E[Transcription & Segmentation]
    C --> F[Remotion Integration]
    F --> G["Shot List Planning (Gemini AI)"]
    E --> G
    G --> H[Final Video Assembly]
`} />

*System architecture: the workflow flows from Script Generation → Audio Processing → Video Generation, with AI modules assisting at each step.*

As a software engineer with only a bit of video editing experience from my days filming weddings in Nepal, I've always found the traditional editing process to be incredibly time-consuming. I often wanted to make quick videos to support different projects, but the manual workload was a huge barrier. Plus, in this era, people are far less likely to read blog posts—video has become a much more engaging format.

The arrival of chatbots like ChatGPT and tools such as Cursor has made it significantly easier and more productive to automate creative tasks with code. Previously, it wasn’t practical to automate editing—I would be trading 4 hours of video editing for 40 hours of building an automation system! But with recent advances, that equation has changed, and building a semi-automated workflow is now far more approachable.

Ultimately, it’s a classic case of “when all you have is a hammer, everything looks like a nail.” For me, coding has become that hammer, and the process of editing video is the nail I set out to automate.

## System Components

### 1. Script Generation & Voiceover Creation

The process begins with creating a compelling voiceover script:

- **Script Creation**: Write the script yourself or use AI tools like ChatGPT
- **Best Practices**: Include detailed notes about the video topic in your prompt
- **Duration**: Currently optimized for 4-5 minute videos
- **Voice Generation**: Use Chatterbox TTS service with your own voice as reference audio

![Screenshot of script generation interface showing AI-generated content with human editing capabilities](placeholder-script-generation.png)

**Technical Challenge**: Chatterbox has a 40-second limit per generation, so I divide the script into smaller paragraphs that fit within this constraint.

### 2. Audio Processing & Transcription

Once the voiceover is generated, it needs to be refined and transcribed:

- **Audio Compilation**: Edit out unnecessary noise and mistakes
- **Segmentation**: Divide into 2-minute segments for optimal processing
- **Transcription**: Use faster_whisper library for word-level and sentence-level timestamps

![Audio waveform visualization showing the segmentation process and timestamp extraction](placeholder-audio-processing.png)

**Custom Solution**: I built my own sentence-level timestamp extractor because the default implementation sometimes cuts sentences in the middle, which breaks my shot list generation.

### 3. AI-Powered Shot List Generation

This is where the magic happens - using AI to plan the visual content:

- **Remotion Integration**: Leverage the JavaScript framework's composition system
- **Available Shots**: Query the `/compositions` endpoint to get available shot types
- **AI Planning**: Send transcript + available compositions to Gemini AI
- **Structured Output**: Receive JSON with sentence timing and composition data

![Screenshot of the shot list generation interface showing AI-generated visual planning](placeholder-shot-list-generation.png)

### 4. Video Assembly & Rendering

The final step brings everything together:

- **MasterSequence Composition**: Converts the JSON shot list into a timeline
- **Integrated Audio**: No need for external NLE editors like Premiere Pro
- **Smart Transitions**: Applied to only 50% of shots to avoid visual overload
- **Direct Rendering**: Export final video from the Remotion timeline

![Final video timeline showing the assembled shots with transitions and audio synchronization](placeholder-video-timeline.png)

## Technical Implementation

The system processes 1-2 minute voiceover segments from a complete 4-6 minute script to generate accurate transcriptions. Each segment goes through:

1. **Whisper Transcription** → Word/sentence timestamps
2. **Custom Sentence Extraction** → Complete sentence boundaries
3. **Gemini AI Planning** → Visual shot recommendations

Here's a visual representation of the workflow:

<Mermaid chart={`graph TD
    A[Voiceover Audio] --> B[Whisper Transcription]
    B --> C[Custom Sentence Extraction]
    C --> D[Gemini AI Planning]
    D --> E[Shot List Generation]
    E --> F[MasterSequence Composition]
    F --> G[Video Rendering]
    
    B --> H[Word/Sentence Timestamps]
    C --> I[Complete Sentence Boundaries]
    D --> J[Visual Shot Recommendations]
    E --> K[JSON Shot List]
    F --> L[Timeline with Transitions]
    G --> M[Final Video]
    
    style A fill:#e1f5fe
    style M fill:#c8e6c9
    style D fill:#fff3e0
    style F fill:#f3e5f5`} 
    
    alt="test"
    figure={true}/>
    
4. **Remotion Rendering** → Final video output

![Technical workflow diagram showing the data flow between different AI services](placeholder-technical-workflow.png)

## Key Benefits

- **Reduced Manual Work**: Automates script-to-video pipeline
- **Consistent Quality**: AI ensures proper timing and visual flow
- **Human Oversight**: Maintains creative control and quality assurance
- **Scalable Process**: Can handle multiple video projects efficiently

## Future Improvements

- Expand beyond 4-5 minute video limit tackling the context limit challenge of llms
- Add more shot types / composotions for visual variety.
- Multi shot sequence for sentences that need them
- Automated image and video retreival and integration.

![Mockup showing potential future features and interface improvements](placeholder-future-features.png)

## Conclusion

This semi-automated system represents a significant step toward more efficient video content creation. While AI handles the technical heavy lifting, human creativity and quality control remain essential for producing engaging content. The combination of modern AI tools with traditional video editing principles creates a powerful workflow that's both efficient and maintainable.

*Have you experimented with AI-powered video creation? I'd love to hear about your experiences and any improvements you've discovered!*