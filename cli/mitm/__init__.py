# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A man-in-the-middle proxy for OpenAI-compatible endpoints.

Fronts an existing OpenAI-compatible server and runs a user-supplied hook against
every chat-completion request. The hook either returns `None`, in which case the
request is forwarded and the client sees exactly what the upstream server would have
sent, or it returns a response, which is framed in the upstream's own protocol and
sent instead.
"""
