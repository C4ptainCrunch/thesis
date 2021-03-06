{% extends 'rst.tpl'%}

{% block any_cell %}
{% if 'ha' in cell['metadata'].get('tags', []) %}
{% elif 'c-intro' in cell['metadata'].get('tags', []) %}
.. raw:: html

      <div class="code-intro">

{{ super()|trim }}

.. raw:: html

      </div>


{% elif 'c-hide' in cell['metadata'].get('tags', []) %}
.. raw:: html

      <div class="code-hide">

{{ super()|trim }}

.. raw:: html

      </div>


{% elif 'pseudocode' in cell['metadata'].get('tags', []) %}
.. raw:: html

      <pre class="pseudocode"  data-controller="pseudocode">
        {{ super()|indent(8, True) }}
      </pre>

{% else %}
  {{ super() }}
{% endif %}
{% endblock any_cell %}

{% block input %}
{% if 'hc' in cell['metadata'].get('tags', []) %}
{% else %}
  {{ super() }}
{% endif %}
{% endblock input %}


{% block stream %}
{% if 'hr' in cell['metadata'].get('tags', []) %}
{% else %}
{{ super() }}
{% endif %}
{% endblock stream %}

{% block data_svg %}

{% for tag in cell['metadata'].get('tags', []) %}
    {% if tag.startswith("fig:") %}
.. _{{tag}}:
    {% endif %}
{% endfor %}

.. figure:: {{ output.metadata.filenames['image/svg+xml'] | urlencode }}
{% endblock data_svg %}
