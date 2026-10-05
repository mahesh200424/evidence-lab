import json
import re
import streamlit as st
import core

st.set_page_config(page_title='Evidence Lab',page_icon='📚',layout='wide')
st.title('Evidence Lab')
st.caption('Local document research · Inspect the evidence · Measure the results')
st.info('Documents and answers stay on this Mac. Models are unloaded after each operation to save RAM. First responses can take longer.')

with st.sidebar:
    st.header('Local models')
    try:
        available=core.models()
    except Exception as exc:
        st.error(str(exc)); st.stop()
    if not available:
        st.error('Install a local chat model in Ollama'); st.stop()
    preferred='phi4-mini:latest'
    model=st.selectbox('Answer model',available,index=available.index(preferred) if preferred in available else 0)
    k=st.slider('Retrieved passages',2,5,4)
    st.caption('Embeddings: nomic-embed-text. Retrieval scores are similarity, not confidence.')
    if st.button('Unload selected model'):
        try:
            core.api('generate',{'model':model,'keep_alive':0}); st.success('Model unloaded')
        except Exception as exc: st.error(str(exc))

library,research,dashboard=st.tabs(['1 · Library','2 · Research','3 · Evaluation'])

with library:
    st.subheader('Build your evidence library')
    st.write('Upload text-based PDFs. Page references use the PDF page position, which may differ from printed page numbers.')
    uploads=st.file_uploader('PDF documents',type=['pdf'],accept_multiple_files=True)
    if st.button('Index uploaded PDFs',disabled=not uploads):
        for file in uploads:
            try:
                with st.spinner(f'Extracting and embedding {file.name}…'):
                    result=core.index_pdf(file.name,file.getvalue())
                st.success(f"{file.name}: "+('already indexed' if result['duplicate'] else f"{result['chunks']} passages indexed"))
            except Exception as exc: st.error(f'{file.name}: {exc}')
    if st.button('Load synthetic demo documents'):
        from demo_data import DOCUMENTS
        for name,pages in DOCUMENTS.items():
            try:
                with st.spinner(f'Indexing {name}…'): core.index_pages(name,pages)
            except Exception as exc: st.error(str(exc)); break
        else: st.success('Demo library ready. These are fictional fixtures, not research findings.')
    docs=core.documents()
    st.dataframe([{'Document':d['name'],'Pages':d['pages'],'Passages':d['chunks']} for d in docs],hide_index=True)
    st.caption('Index and uploaded PDFs persist in data/. Scanned PDFs, diagrams, and complex tables are not interpreted in this version.')


def show_answer(result):
    st.subheader('Answer')
    st.write(result['answer'])
    if result['citations']:
        st.markdown('Sources: '+ ' · '.join(f'[{c}](#evidence-{c.lower()})' for c in dict.fromkeys(result['citations'])))
    st.caption(f"{result['model']} · {result['seconds']} s · {result['output_tokens']} output tokens")
    if result['citation_check_rejected']: st.warning('The model failed the citation-ID check. Its answer was withheld.')
    st.caption('Valid citation IDs do not prove factual support. Inspect the passages below.')
    st.subheader('Evidence')
    for s in result['sources']:
        st.markdown(f'<div id="evidence-{s["citation"].lower()}"></div>',unsafe_allow_html=True)
        cited=' · cited' if s['citation'] in result['citations'] else ' · retrieved only'
        with st.expander(f"[{s['citation']}] {s['name']} — page {s['page']}{cited}",expanded=s['citation'] in result['citations']):
            st.write(s['text']); st.caption(f"Cosine similarity: {s['score']}")
            if s['file']:
                st.download_button('Download source PDF',(core.DATA/s['file']).read_bytes(),file_name=s['name'],mime='application/pdf',key='source-'+s['id'])

with research:
    st.subheader('Ask your documents')
    question=st.text_area('Question',placeholder='What does the document say about retrieval evaluation?',max_chars=2000)
    a,b=st.columns(2)
    if a.button('Inspect retrieval',disabled=not question.strip()):
        try:
            with st.spinner('Searching local passages…'): st.session_state['retrieved']=core.retrieve(question,k)
            st.session_state.pop('answer',None)
        except Exception as exc: st.error(str(exc))
    if b.button('Generate cited answer',type='primary',disabled=not question.strip()):
        try:
            with st.spinner('Retrieving evidence and generating locally…'): st.session_state['answer']=core.answer(question,model,k)
            st.session_state.pop('retrieved',None)
        except Exception as exc: st.error(str(exc))
    if 'answer' in st.session_state: show_answer(st.session_state['answer'])
    elif 'retrieved' in st.session_state:
        for s in st.session_state['retrieved']:
            with st.expander(f"{s['citation']} · {s['name']} · page {s['page']}",expanded=True):
                st.write(s['text']); st.caption(f"Similarity {s['score']}")

with dashboard:
    st.subheader('Evaluate, compare, and review')
    st.write('Use the same labelled questions and corpus when comparing runs. Automatic checks measure retrieval and abstention; correctness and citation support require your review.')
    from demo_data import CASES
    st.download_button('Download demo evaluation JSON',json.dumps(CASES,indent=2),'evaluation-demo.json','application/json')
    dataset=st.file_uploader('Custom evaluation JSON',type=['json'])
    use_demo=st.checkbox('Use demo questions (load demo documents first)',value=True)
    if st.button('Run evaluation',disabled=not (use_demo or dataset)):
        try:
            cases=CASES if use_demo else json.loads(dataset.getvalue())
            if use_demo and not {'demo-retrieval.txt','demo-models.txt'}.issubset({d['name'] for d in core.documents()}):
                raise ValueError('Load the synthetic demo documents in Library first')
            progress=st.progress(0,text='Starting evaluation…')
            with st.spinner('Running local evaluation. This may take several minutes.'):
                run_id,results=core.evaluate(cases,model,k,lambda i,n:progress.progress(i/n,text=f'Question {i+1}/{n}'))
            progress.progress(1.,text=f'Run {run_id} saved')
        except Exception as exc: st.error(str(exc))
    saved=core.runs()
    if saved:
        summary=[]
        for run in saved:
            summary.append({'Run':run['id'],'Configuration':run['label'],'Created':run['created'],**core.metrics(json.loads(run['results']))})
        st.dataframe(summary,hide_index=True)
        selected=st.selectbox('Inspect and review a run',[r['id'] for r in saved])
        run=next(r for r in saved if r['id']==selected)
        results=json.loads(run['results'])
        st.download_button('Export complete run',json.dumps(results,indent=2),f'run-{selected}.json','application/json')
        for i,result in enumerate(results):
            with st.expander(f"{i+1}. {result['question']}"):
                if 'error' in result: st.error(result['error']); continue
                st.write(result['answer'])
                st.write('Expected answer:',result['expected_answer'])
                st.write('Retrieval hit:',result['retrieval_hit'],'Abstention behaviour correct:',result['abstention_correct'])
                for source in result['sources']:
                    st.write(f"[{source['citation']}] {source['name']} p{source['page']}: {source['text']}")
                for field,label in [('human_correct','Is the answer correct?'),('human_supported','Do the citations support all factual claims?')]:
                    choices=['Not reviewed','Yes','No']; values=[None,True,False]
                    choice=st.selectbox(label,choices,index=values.index(result.get(field)),key=f'{selected}-{i}-{field}')
                    result[field]=values[choices.index(choice)]
        if st.button('Save human reviews'):
            core.save_review(selected,results); st.rerun()
    else: st.caption('No evaluation runs yet.')
