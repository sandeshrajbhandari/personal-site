---
title: Testing Mermaid Diagram Support
date: '2023-10-23'
tags: ['test', 'mermaid']
draft: False
summary: Testing Mermaid diagram integration in blog posts
---

This is a test blog post to verify that Mermaid diagrams work correctly in blog posts.

## Simple Flowchart

Here's a basic flowchart:

```mermaid
graph TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Great!]
    B -->|No| D[Debug]
    D --> B
```

## Sequence Diagram

Here's a sequence diagram showing the process:

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant Server
    participant Database

    User->>Browser: Request Page
    Browser->>Server: HTTP Request
    Server->>Database: Query Data
    Database-->>Server: Return Data
    Server-->>Browser: HTML Response
    Browser-->>User: Display Page
```

## Class Diagram

Here's a simple class diagram:

```mermaid
classDiagram
    class Animal {
        +String name
        +int age
        +makeSound()
    }

    class Dog {
        +bark()
    }

    class Cat {
        +meow()
    }

    Animal <|-- Dog
    Animal <|-- Cat
```

## Gantt Chart

Here's a project timeline:

```mermaid
gantt
    title Project Timeline
    dateFormat  YYYY-MM-DD
    section Planning
    Research       :done,    des1, 2023-10-01, 2023-10-05
    Design          :done,    des2, after des1, 3d
    section Development
    Coding          :active,  dev1, 2023-10-10, 7d
    Testing         :         dev2, after dev1, 3d
    section Deployment
    Documentation  :         doc1, after dev2, 2d
    Deployment      :         dep1, after doc1, 1d
```

## Pie Chart

Here's a data visualization:

```mermaid
pie
    title Technology Usage
    "JavaScript" : 35
    "Python" : 25
    "React" : 20
    "Node.js" : 15
    "Other" : 5
```

This demonstrates that various types of Mermaid diagrams should render properly in blog posts!