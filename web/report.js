function createRecord(){
  const date=new Date();
  return {id:'QA-'+date.toISOString().slice(0,10).replaceAll('-','')+'-'+crypto.randomUUID().slice(0,8).toUpperCase(),opened:date.toLocaleDateString('en-GB'),revision:0};
}
function reportHeader(title,form){
  return '<div class="formhead"><div><div class="formcode">QUALITY ASSURANCE &nbsp; / &nbsp; '+form+'</div><h2>'+title+'</h2></div><strong>DRAFT<br><small>Engineer review</small></strong></div>';
}
function block(title,body){return '<div class="band">'+title+'</div>'+body}
function actionTable(rows){
  return table(['What','Where','When','Who','Required Results'],rows);
}
function caseForms(ranked,remaining){
  const family=current.family.replaceAll('_',' '), lead=ranked[0];
  const info=table(['Tracking number','Opened','Revision','Case status'],[[record.id,record.opened,String(record.revision).padStart(2,'0'),'Open · investigation']]);
  const process='Originating operation to be established';
  const causes=ranked.map(row=>[family,row.potential_effect,row.candidate_cause,row.prevention,row.detection,'—','—','—']);
  const draft=current.draft;
  const actions=type=>draft.actions.filter(a=>a.type===type).map(a=>[a.what,a.where,a.when,a.who,a.required_results]);
  const evidenceRows=history.map((h,i)=>['E'+(i+1),scenario.tests.find(t=>t.id===h.test).label,h.outcome.replaceAll('_',' '),h.source]);
  const check=remaining[0];
  const next=check?check.label:'Review recorded findings and any unresolved alternatives.';
  const pfmea='<section class="sheet">'+reportHeader('Process FMEA worksheet','PFMEA')+info+
    block('Scope and process function',table(['Process step','Product requirement','Reported failure'],[[process,'Obtain applicable product specification and acceptance limit.',description]]))+
    block('Failure analysis and current controls',table(['Failure mode','Potential effect','Potential cause','Prevention control','Detection control','S','O','D'],causes))+
    '<p class="formnote">S = severity · O = occurrence · D = detection. A dash marks a rating awaiting evidence and the plant scoring criteria. Priority calculation is deferred.</p>'+
    block('Review assignment',table(['Process owner','FMEA reviewer','Review date'],[['Unassigned','Pending assignment','Pending review']]))+'</section>';
  const capa='<section class="sheet">'+reportHeader('Corrective & preventive action report','CAPA')+info+
    block('01 &nbsp; Case identification',table(['Submitted by','Issue source','Affected lot / coil','Investigation team'],[['Not supplied','To be confirmed','Not supplied','To be assigned']]))+
    block('02 &nbsp; Issue as reported','<p class="statement">'+esc(description)+'</p>')+
    block('03 &nbsp; Immediate containment · proposed',actionTable(actions('containment')))+
    block('04 &nbsp; Root cause investigation',table(['Cause status','Lead hypothesis under current assumptions','Next investigation step'],[['Cause unconfirmed',lead.candidate_cause,next]])+'<p class=statement>'+esc(draft.investigation)+'</p>')+
    block('05 &nbsp; Corrective action plan · conditional',actionTable(actions('corrective')))+
    block('06 &nbsp; Preventive action plan · conditional',actionTable(actions('preventive')))+
    '<p class="formnote">¹ Suggested role. A named assignee and committed due date require supervisor review.</p>'+
    block('07 &nbsp; Effectiveness and closure',table(['Verification plan','Acceptance criterion','Actual result'],[[draft.verification+' '+draft.actions.filter(a=>a.type==='verification').map(a=>a.what+' — '+a.required_results).join(' '),'Engineer to agree the threshold and observation window before verification.','Pending follow-up']])+
    table(['Implementation approved by / date','Effectiveness verified by / date','Closure date','Related records'],[['Pending','Pending','Open','Pending document references']]))+'</section>';
  const annex='<section class="sheet">'+reportHeader('Investigation evidence','CASE ANNEX')+info+
    block('Inspection records',evidenceRows.length?table(['Ref.','Inspection','Finding','Source record'],evidenceRows):'<p class="statement">No inspection finding recorded.</p>')+
    block('Diagnostic ranking',table(['Candidate cause','Initial weight','Current weight'],ranked.map(r=>[r.candidate_cause,(100*r.prior_weight).toFixed(1)+'%',(100*r.current_weight).toFixed(1)+'%'])))+
    '<p class="formnote">Computed with authored scenario estimates. These weights describe relative support among modeled causes. FMEA occurrence ratings require separate evidence.</p>'+
    '<details class="formnote"><summary>Report sources and unresolved information</summary>'+
    (current.references.length?table(['Source ID','Reference','Location','Representation'],current.references.map(r=>[r.id,r.title,r.locator,r.representation])):'<p>No eligible checked reference met the relevance threshold. Authored scenario knowledge was supplied separately.</p>')+
    '<p>'+esc(draft.unresolved.join(' · '))+'</p><p>Generation model: '+esc(current.model)+'. Structure and reference IDs checked; engineering facts require review.</p></details></section>';
  return '<div class="controls reportbar"><span>Case documents · '+esc(record.id)+'</span><button id=printReport>Print report / save PDF</button></div>'+pfmea+capa+annex;
}
