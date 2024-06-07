---
title: Why you need a personal website - Hello World
date: '2022-05-21'
tags: ["firestore", "nosql", "filter", "google cloud", "firestore-query", "backend"]
draft: false
summary: "Learn how to implement substring search and combine filters using OR and AND operations for efficient data retrieval from Firestore. This blog covers code examples, best practices, and workarounds for common challenges when working with Firestore queries in Python, including a lack of global search functionality."
---

I'm writing this because there was no documentation reference or example code that showed how to perform complex firestore filter queries in python. I had to write these queries to write a substring search across a range of string and number fields in firestore. there is no global search feature in firestore, so this was a hack to enable substring search across relevant fields across documents in our collection. 

this blog will introduce code to do a substring search in a single field and use that to perform search across mutliple fields by doing a combined query by writing filters and combining them with OR and AND operations in firestore. 

substring search code:

[stackoverflow link](https://stackoverflow.com/questions/46568142/google-firestore-query-on-substring-of-a-property-value-text-search)

```python
collectionRef
    .where('name', '>=', queryText)
    .where('name', '<=', queryText+ '\uf8ff')
```

writing a simple firestore AND filter :

This part adds to the official documentation for using Or and And filters in a combined query.

https://cloud.google.com/firestore/docs/query-data/queries#compound_and_queries

```python
from google.cloud.firestore_v1.base_query import FieldFilter, Or, And

andFilterList = [
        firestore.And(
            [
                FieldFilter(
                    "first_name", ">=", query
                ),
                FieldFilter(
                    "first_name", "<=", query + "\uf8ff"
                ),
            ]
        ),
        firestore.And(
            [
                FieldFilter(
                    "last_name", ">=", query
                ),
                FieldFilter(
                    "last_name", "<=", query + "\uf8ff"
                ),
            ]
        ),        
 ]
 
 combined_query = coll_ref.where(filter=Or(leads_filter_list)).stream()
```

The code uses a list of filters which can be fed to the Or filter. each filter in the list is a result of an `And` filter. There is no offical doc mentioning this particular syntax, but this approach seems to work. The `firestore.And` function works but not the `And` imported from the `base_query`.

each And filter checks if the query is a substring of first_name and last_name.

then the filter is fed to the Or filter, so that if one of the two And filters is true, it returns true and fetches the matching documents. 

One small caveat here is that the substring query technique use here only works for matching first letters of queries and fields. for eg. 

query = jo, first_name = Joanne, Will Work
query = nne, first_name = Joanne, NOT Work

a simpler And implementation would look like this.

```python
and_filter = firestore.And([
	FieldFilter("age", ">", 20),
	FieldFilter("salary", ">" 20000)
])
and_query = coll_ref.where(filter=and_filter).stream()
```
