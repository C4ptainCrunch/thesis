{% extends 'rst.tpl'%}

{% block any_cell %}
{% if 'ha' in cell['metadata'].get('tags', []) %}
{% else %}
    {{ super() }}
{% endif %}
{% endblock any_cell %}


{% block execute_result %}
{% if 'hr' in cell['metadata'].get('tags', []) %}
{% else %}
    {{ super() }}
{% endif %}
{% endblock execute_result %}
