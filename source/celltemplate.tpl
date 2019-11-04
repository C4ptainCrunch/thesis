{% extends 'rst.tpl'%}

{% block any_cell %}
{% if 'ha' in cell['metadata'].get('tags', []) %}
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


{% block execute_result %}
{% if 'hr' in cell['metadata'].get('tags', []) %}
{% else %}
    {{ super() }}
{% endif %}
{% endblock execute_result %}
