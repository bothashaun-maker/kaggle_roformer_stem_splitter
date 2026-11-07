import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

let newest=null;
async function refresh(){
 try{
  const response=await fetch('/stem/songs',{cache:'no-store'});if(!response.ok)return;
  const songs=await response.json();
  for(const node of app.graph?._nodes||[]){
   if(node.type!=='SSLoadSong')continue;
   if(!node.widgets?.some(w=>w.name==='Upload song'))node.addWidget('button','Upload song',null,()=>window.open('/stem/upload','_blank'),{serialize:false});
   const w=node.widgets?.find(w=>w.name==='audio');if(!w)continue;
   w.options.values=songs.length?songs:['(upload a song first)'];
   if(newest&&songs.includes(newest))w.value=newest;
   else if(!w.options.values.includes(w.value))w.value=w.options.values[0];
   node.graph?.change();
   node.graph?.setDirtyCanvas(true,true);
  }
  newest=null;
 }catch(error){console.warn('Stem Studio song list',error);}
}
app.registerExtension({
 name:'StemStudio.Canvas',
 beforeRegisterNodeDef(type,data){
  if(data.name!=='SSExport')return;
  const original=type.prototype.onExecuted;
  type.prototype.onExecuted=function(message){
   original?.apply(this,arguments);
   if(message.stem_download?.[0]){
    this._stemJob=message.stem_download[0];
    if(!this._stemButton)this._stemButton=this.addWidget('button','Download all stems ZIP',null,()=>window.open('/stem/download/'+encodeURIComponent(this._stemJob),'_blank'),{serialize:false});
   }
   if(Array.isArray(message.text)){
    if(!this._stemReport){this._stemReport=ComfyWidgets.STRING(this,'result_report',['STRING',{multiline:true}],app).widget;this._stemReport.inputEl.readOnly=true;this._stemReport.options={...this._stemReport.options,serialize:false};}
    this._stemReport.value=message.text.join('\n');this.setSize([400,260]);
   }
  };
 },
 setup(){
  const url=new URL(location.href),key=url.searchParams.get('stem_studio');
  if(['six','four','vocals'].includes(key))void(async()=>{
   try{
    const deadline=Date.now()+60000;
   const ready=()=>{try{return !!app.graph&&!!app.canvas?.canvas;}catch{return false;}};
   // In the pinned frontend, workspace.spinner stays true through initial
   // workflow/tab restoration (GraphCanvas.vue). Opening earlier lets its
   // default image graph overwrite the requested audio canvas afterward.
   while(!app.vueAppReady||!app.positionConversion||!ready()||app.extensionManager?.spinner||!window.LiteGraph?.registered_node_types?.SSSeparate){if(Date.now()>deadline)throw new Error('ComfyUI did not finish starting');await new Promise(r=>setTimeout(r,100));}
    await new Promise(r=>setTimeout(r,500));
    const r=await fetch(new URL(key+'.json',import.meta.url),{cache:'no-store'});if(!r.ok)throw new Error('Canvas could not be loaded');
    await app.loadGraphData(await r.json(),true,true,'Stem Studio - '+key+'.json');
    // loadGraphData changes the current workflow hash. Preserve that new hash
    // and Vue Router's history state when removing our one-time query flag.
    const clean=new URL(location.href);clean.searchParams.delete('stem_studio');
    history.replaceState(history.state,'',clean);await refresh();
   }catch(error){window.alert('Stem Studio: '+error.message);}
  })();
  try{new BroadcastChannel('stem-songs').onmessage=e=>{newest=e.data.filename;void refresh();};}catch{}
  setInterval(()=>void refresh(),5000);
 },
 nodeCreated(node){
  if(node.type==='SSLoadSong'){
   setTimeout(()=>void refresh(),300);
  }
 }
});
