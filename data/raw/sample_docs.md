# Authentication Guide

## Overview
This guide explains how JWT tokens work.

## Token Refresh
Tokens expire after 60 minutes. Use the refresh endpoint to obtain a new token.

`python
def refresh():
    return 'new_token'
``n
