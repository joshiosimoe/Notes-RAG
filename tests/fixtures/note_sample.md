---
title: Kubernetes Scheduling
tags: [kubernetes, scheduling]
status: reviewed
---

Intro paragraph before any heading. Mentions [[Bloom Filters]] in passing.

## Filtering phase

The scheduler removes nodes that cannot host the pod. Related: [[Node Affinity|affinity rules]].

## Scoring phase

Remaining nodes are ranked. See [[Scoring#Priorities]] for the weight table.
