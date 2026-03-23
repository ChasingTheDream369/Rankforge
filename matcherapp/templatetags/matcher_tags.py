from django import template

register = template.Library()


@register.filter
def pct(value):
    try:
        return f'{float(value):.0%}'
    except (TypeError, ValueError):
        return '0%'


@register.filter
def score_color(value):
    try:
        v = float(value)
        if v >= 0.70:
            return 'success'
        if v >= 0.40:
            return 'warning'
        return 'error'
    except (TypeError, ValueError):
        return 'error'


@register.filter
def split(value, delimiter=','):
    return str(value).split(delimiter)


@register.filter
def rec_label(value):
    labels = {
        'STRONG_MATCH': 'Strong Match',
        'GOOD_MATCH': 'Good Match',
        'PARTIAL_MATCH': 'Partial Match',
        'WEAK_MATCH': 'Weak Match',
        'NO_MATCH': 'No Match',
    }
    return labels.get(str(value), value)


@register.filter
def skill_label(skill):
    """Unified label for skill dicts: jd_skill (D1 profile) or skill (deterministic)."""
    if not isinstance(skill, dict):
        return ''
    return skill.get('jd_skill') or skill.get('skill', '')


@register.filter
def skill_badge(skill):
    """Unified badge for skill dicts: match_type (D1 profile) or level (deterministic)."""
    if not isinstance(skill, dict):
        return ''
    return skill.get('match_type') or skill.get('level') or ''


@register.filter
def constraint_label(check):
    """Unified label for constraint dicts: constraint (D4 profile) or req (deterministic)."""
    if not isinstance(check, dict):
        return ''
    return check.get('constraint') or check.get('req', '')


@register.filter
def constraint_status(check):
    """
    Returns 'met', 'partial', or 'not_met' for constraint dicts.
    Profile format: {constraint, score} with score 1.0|0.5|0.0
    Deterministic format: {req, met, partial}
    """
    if not isinstance(check, dict):
        return 'not_met'
    score = check.get('score')
    if score is not None:
        if score >= 1.0:
            return 'met'
        if score >= 0.5:
            return 'partial'
        return 'not_met'
    if check.get('met'):
        return 'met'
    if check.get('partial'):
        return 'partial'
    return 'not_met'


@register.filter
def constraint_display(check):
    """Display string for constraint status: 100%/50%/0% or Met/Partial/Not met."""
    if not isinstance(check, dict):
        return 'Not met'
    score = check.get('score')
    if score is not None:
        return f'{int(score * 100)}%'
    if check.get('met'):
        return 'Met'
    if check.get('partial'):
        return 'Partial'
    return 'Not met'
