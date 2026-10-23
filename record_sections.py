"""Route whole observed replies by the interview topic, without generating new facts."""
import re

# Adapted from the source-linked routing of the 1020_v1 archive; no SOAP review loop.
TOPIC_SECTION={'concern':'hpi','onset':'hpi','course':'hpi','associated_1':'hpi',
    'associated_2':'hpi','red_flags':'ros','impact':'hpi','prior_care':'hpi',
    'past':'pmh','medications':'pmh','allergy':'pmh','birth':'pmh','development':'pmh',
    'menstrual':'pmh','pregnancy':'pmh','treatment_safety':'ros','family':'fh',
    'family_course':'fh','family_current':'fh','smoking':'sh','alcohol':'sh',
    'occupation':'sh','daily_life':'sh','med_name':'pmh','med_strength':'pmh','med_sources':'pmh',
    'severity':'hpi','pattern':'hpi','triggers':'hpi','associated_3':'hpi','ros_1':'ros','ros_2':'ros','mood':'ros',
    'ros_general':'ros','ros_sleep':'ros','lifestyle':'sh','expectation':'hpi'}


def topic_section(topic, by_turn):
    if topic.startswith('clarify:'):
        return topic_section(topic.split(':')[1],by_turn)
    if topic.startswith('followup:'):
        return by_turn.get(topic.split(':',1)[1],'hpi')
    if topic in ('med_name','med_strength','med_sources'):
        return 'pmh'
    if topic.startswith('med_') or topic=='complete:initial_amount':
        return 'hpi'
    return TOPIC_SECTION.get(topic,'ros')


def organize_record(note, items, history):
    """Keep the original CC and all raw source values; change placement, not their meaning."""
    by_turn={}
    for row in history:
        section=topic_section(row['topic'],by_turn)
        # A generated generic topic label must not turn symptom review into social history.
        if row['topic']=='daily_life' and re.search(r'떨림|떨리|두근|두통|통증|숨|설사|변비',row.get('question','')):
            section='ros'
        by_turn[row['source_id']]=section
    fields={key:[] for key in ('hpi','pmh','fh','sh','ros')};exams=[]
    def line(item):
        marker='V/S' if item['id']=='vitals' else item['id'].replace('turn','턴 ')
        question=item.get('question','')
        return f'[{marker}] '+(question+' → ' if question else '')+item['value']
    for item in items:
        if item['kind']=='opening':
            fields['hpi'].append(line(item))
        elif item['kind']=='exam':
            exams.append(line(item))
        elif item['kind']=='say':
            fields[by_turn.get(item['id'],'hpi')].append(line(item))
    for key,values in fields.items():
        note['S'][key]='\n'.join(values)
    note['O']['exam']='\n'.join(exams)
    note['O']['tests']=''
    return note


def connect_followup_warnings(note):
    """Copy already-planned warning/action sentences; never create a new clinical recommendation."""
    prospective=note['P']['education'].split('\n[실제')[0].removeprefix('[계획/제안, 시행 결과 아님] ')
    clauses=re.split(r'(?<=[.!?。])\s+|\n+',prospective)
    linked=[]
    for clause in clauses:
        clause=clause.strip()
        if (clause and re.search(r'즉시|응급|바로|urgent|emergency|immediate',clause,re.I)
                and re.search(r'경우|생기|발생|하면|시에는|이면|때는|\bif\b|\bwhen\b|requires?|warrants?',clause,re.I)
                and clause not in note['P']['follow_up']):
            note['P']['follow_up']+='\n'+clause
            linked.append(clause)
    return linked
