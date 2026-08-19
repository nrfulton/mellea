# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A man-in-the-middle proxy for OpenAI-compatible endpoints.

Fronts an existing OpenAI-compatible server and runs a user-supplied hook against
every chat-completion request. The hook either returns `None`, in which case the
request is forwarded and the client sees exactly what the upstream server would have
sent, or it returns a response, which is framed in the upstream's own protocol and
sent instead.

A second hook may be run against the reply the upstream produced, for anything that
can only be judged once there is a reply to judge. `policy` implements that case for
behavioural policies in the `granite.trust.policy-tools` YAML format, whose
restrictions describe what a reply must not contain.
"""
