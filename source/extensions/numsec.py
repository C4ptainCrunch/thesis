# Copyright (c) 2012, Jeff Terrace
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""
Changes section references to be the section number
instead of the title of the section.
"""

from docutils import nodes
from sphinx.domains import Domain
from sphinx.roles import XRefRole

class NumSecDomain(Domain):

    name = 'numsec'
    label = 'Num sec'
    roles = {
        'ref': XRefRole()
    }

    def resolve_xref(self, env, fromdocname, builder,
                     typ, target, node, contnode):
        res = super().resolve_xref(env, fromdocname, builder,
                                                            typ, target, node, contnode)

        if res is None:
            return res

        if typ == 'ref' and not node['refexplicit']:
            docname, labelid, sectname = self.data['labels'].get(target, ('','',''))
            res['refdocname'] = docname

        return res

def doctree_resolved(app, doctree, docname):
    secnums = app.builder.env.toc_secnumbers
    import ipdb; ipdb.set_trace()

    for node in doctree.traverse(nodes.reference):
        # print(node, "\n")
        if 'refdocname' in node:
            refdocname = node['refdocname']
            if refdocname in secnums:
                secnum = secnums[refdocname]
                toclist = app.builder.env.tocs[refdocname]
                for child in node.traverse():
                    if isinstance(child, nodes.Text):
                        text = child.astext()
                        print(text)
                        anchorname = None
                        for refnode in toclist.traverse(nodes.reference):
                            if refnode.astext() == text:
                                anchorname = refnode['anchorname']
                        if anchorname is None:
                            continue
                        linktext = '.'.join(map(str, secnum[anchorname]))
                        child.parent.replace(child, nodes.Text(linktext))

def setup(app):
    app.add_domain(NumSecDomain, override=True)
    app.connect('doctree-resolved', doctree_resolved)