---
title: How I use Obsidian and Anki to remember things
date: '2022-05-21'
tags: ['anki', 'digital-garden', 'workflow']
draft: true
summary: 'Anki is a powerful learning tool that uses spaced-repitition to remeber things long term. Using default Anki interface to create flashcards is a hassle, so I use Obsidian to create question answers in bulk and transfer them to Anki in one click using a powerful community plugin.'
---

Learning new things is powerful in this age of new technologies. I usually explore and learn the latest software and tools being developed in AI and machine learning field to see if there is anything I can use in my day to day life. However, with the excess of information, I seem to struggle with remembering what I have learned in the past. To solve this problem, I started looking into different learning techniques. Anki came out as a reliable means to do this. However, preparing anki notes is a hassle with Anki's interface, especially as the number of flashcards grow. So, I looked into easing this process.

## Installing Obsidian to Anki

Please refer to the docs for installing the plugin for obsidian and configuring Anki to work with the plugin
https://github.com/Pseudonium/Obsidian_to_Anki
After you configure it, you can prepare notes in the required format.
You can change the format of the notes to convert to Anki flashcards with Regex code in plugin settings.

## Obsidian to Anki

At first I prepare question answers in Obsidian, sometimes by scraping from different sources, and sometimes by manually typing them in. By default Anki has an interface to add flashcards one by one, but this takes more time. Obsidian has a commmunity plugin that lets you transfer notes in specific formats to Anki as flashcards. I use the following Heading format, but you can explore more formats in the plugin docs.https://github.com/Pseudonium/Obsidian_to_Anki

```md
#### What is the height of Mt. Everest?

8848m
```

use the following format to specify tags and target deck for the flashcards in the text file.

```
FILE TAGS: tag1 tag2
TARGET DECK: Deck1
```

if the target deck is not present in Anki, it creates the deck when you first click the Obsidian-to-anki icon. then you have to reclick it. nothing happens when you reclick it because there is no change in text file. so press enter or add/remove text and then click the Obsidian-to-anki icon. On successful creation of the flashcards from the Obsidian note, you will see the commented id of the note.`<!--ID: 1660055697480-->`
here is a sample text file after flashcards are made from the notes.

```
FILE TAGS: tag1 tag2
TARGET DECK: Deck1

#### What is the height of Mt. Everest?
8848m
<!--ID: 1660055697480-->
```

quality of life hack, so adding hashes infront of each question is a hassle, so I write the notes without them. then I run a python script to add 4 hashes infront of each line with a question mark in it. then I export them to Anki. The code reads from a text file and copies the modified text with heading/hashes into the clipboard. I paste it to obsidian.

```python
import codecs, pyperclip
outputString = ""
with codecs.open('readme.txt', encoding='utf-8') as file:
#with open('readme.txt', 'r') as file:
    for line in file:
        #print (line[len(line)-3])
        if '?' in line:
            outputString = outputString + "##### " + line
            #print('#' + line)
        else:
            outputString = outputString + line
            #print(line)
    pyperclip.copy(outputString)
    #print ("Output: \n" + outputString)
```
