'use strict';
const $=id=>document.getElementById(id),esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data=null,current=null,scenario=null,description='',history=[],record=null,token='',busy=false;
function table(headers,rows){return '<div class="scroll"><table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function message(text,error=false){$('status').hidden=!text;$('status').textContent=text;$('status').style.color=error?'#922f32':'';if($('inspectionMessage'))$('inspectionMessage').textContent=text;}
function lock(value){busy=value;for(const id of ['run','example','apply','regenerate','confirmFamily','networkYes'])if($(id))$(id).disabled=value;}
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json','X-Case-Token':token},...(body?{body:JSON.stringify(body)}:{})});const d=await r.json();if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:'Please check the entered information.');return d;}
function remember(){sessionStorage.setItem('steelOriginalDesk',JSON.stringify({id:data.id,token}));}
function hydrate(){
 description=data.description;history=data.history.map(h=>({test:h.test_id,outcome:h.outcome,source:h.source||''}));
 scenario={tests:data.tests_catalog};
 record={id:data.tracking,opened:data.opened,revision:data.revision};
 const generated=Object.fromEntries((data.draft?.fmea||[]).map(r=>[r.cause_id,r]));
 current={family:data.family,draft:data.draft,references:data.references,model:data.model,fmea:data.ranking.map(r=>({id:r.id,candidate_cause:r.label,prior_weight:r.initial,current_weight:r.current,potential_effect:generated[r.id]?.effect||'Pending generation',prevention:generated[r.id]?.prevention||'To be verified',detection:generated[r.id]?.detection||'To be verified'}))};
}
function render(){
 hydrate();const ranked=[...current.fmea].sort((a,b)=>b.current_weight-a.current_weight);
 const remaining=[...data.tests].sort((a,b)=>Number(b.id===data.recommended_test)-Number(a.id===data.recommended_test));
 let html=data.draft?caseForms(ranked,remaining):'<section><p>The investigation is saved. FMEA and CAPA reports will appear after generation completes.</p></section>';
 const stale=data.draft&&data.draft_revision!==data.revision;
 html+='<section class=controls><h2>Inspection evidence</h2>'+history.map(h=>'<p>'+esc(scenario.tests.find(t=>t.id===h.test)?.label||h.test)+' · '+esc(h.outcome.replaceAll('_',' '))+' · '+esc(h.source)+'</p>').join('');
 if(remaining.length)html+='<label for=test>Inspection</label><select id=test>'+remaining.map(t=>'<option value="'+esc(t.id)+'">'+esc(t.label)+'</option>').join('')+'</select><label for=out>Finding</label><select id=out></select><label for=source>Inspection record or source</label><textarea id=source rows=1></textarea><button id=apply>Update assessment</button>';
 else html+='<p>All available inspections recorded. Review the evidence and any unresolved alternatives.</p>';
 html+='<p id=inspectionMessage role=status aria-live=polite></p>'+(!data.draft||stale?'<button class=alt id=regenerate>Retry report generation</button>':'')+'</section>';
 $('result').innerHTML=html;
 if($('printReport')){$('printReport').disabled=!!stale;$('printReport').onclick=()=>window.print();}
 if(stale)message('New findings are saved. The displayed report predates them; regenerate before printing.',true);
 if(remaining.length){const outcomes=()=>{const t=remaining.find(t=>t.id===$('test').value);$('out').innerHTML='<option value="">Select the observed finding</option>'+t.outcomes.map(o=>'<option value="'+esc(o)+'">'+esc(o.replaceAll('_',' '))+'</option>').join('');};$('test').onchange=outcomes;outcomes();$('apply').onclick=recordFinding;}
 if($('regenerate'))$('regenerate').onclick=async()=>{if(busy)return;lock(true);try{await generateReports();}finally{lock(false);}};
 lock(busy);
}
async function generateReports(){
 message('Retrieving relevant knowledge and generating FMEA and CAPA…');
 try{data=await api('/api/cases/'+data.id+'/generate',{revision:data.revision});remember();render();message('Reports ready for engineer review.');return true;}
 catch(e){render();message(e.message,true);return false;}
}
async function generate(family=null){
 if(busy)return;const text=$('desc').value.trim();if(text.length<3){message('Enter what you observed on the material.',true);return;}
 lock(true);message('Interpreting the observation…');
 try{
 const answer=await api('/api/cases',{description:text,family});
 if(answer.clarify){
  message('');
  if(answer.clarify==='cracks'){$('result').innerHTML='<section><h2>Surface cracking recognized</h2><p>'+esc(answer.message)+'</p><button id=networkYes>Yes · fine crack network</button><button class=alt id=networkNo>No / unsure</button><p id=crackHelp></p></section>';$('networkYes').onclick=()=>generate('Crazing');$('networkNo').onclick=()=>{$('crackHelp').textContent='Please add the crack direction, location and pattern, or seek a quality engineer assessment. Isolated or edge cracks must not be forced into Crazing.';};}
  else{$('result').innerHTML='<section><h2>Please confirm the observation</h2><p>'+esc(answer.message)+'</p><select id=pick><option value="">Select a defect family…</option>'+answer.families.map(f=>'<option value="'+esc(f)+'">'+esc(f.replaceAll('_',' '))+'</option>').join('')+'</select><button id=confirmFamily>Confirm observed defect</button></section>';$('confirmFamily').onclick=()=>{if($('pick').value)generate($('pick').value);};}
  return;
 }
 data=answer.case;token=answer.token;remember();render();await generateReports();
 }catch(e){message(e.message,true);}finally{lock(false);}
}
async function recordFinding(){
 if(busy)return;const outcome=$('out').value,source=$('source').value.trim();
 if(!outcome||source.length<2){message('Select a finding and enter the inspection record or source.',true);return;}
 const y=window.scrollY;lock(true);
 try{data=await api('/api/cases/'+data.id+'/findings',{test_id:$('test').value,outcome,source,revision:data.revision});remember();render();message('Finding saved. Bayesian weights updated. Refreshing reports…');await generateReports();window.scrollTo(0,y);}
 catch(e){message(e.message,true);}finally{lock(false);}
}
$('run').onclick=()=>generate();$('example').onclick=()=>{$('desc').value='Longitudinal scratches on a cold-rolled coil at final inspection.';generate();};
(async()=>{try{const saved=JSON.parse(sessionStorage.getItem('steelOriginalDesk'));if(saved){token=saved.token;data=await api('/api/cases/'+saved.id);$('desc').value=data.description;render();message('Previous investigation restored.');}}catch{sessionStorage.removeItem('steelOriginalDesk');}})();
