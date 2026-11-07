/* Run against a local test server. Uses synthetic media only. */
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const base=process.env.STEM_TEST_URL||'http://127.0.0.1:8399';
const root=process.env.STEM_TEST_FILES||'/tmp/stem-research';
(async()=>{
 const deadline=Date.now()+45000;
 while(true){
  try{const r=await fetch(base+'/object_info',{signal:AbortSignal.timeout(3000)});if(r.ok)break;}catch{}
  if(Date.now()>deadline)throw new Error('Test ComfyUI server did not become ready');
  await new Promise(r=>setTimeout(r,500));
 }
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1500,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/?stem_studio=six');
  // ComfyUI's compatibility module snapshots a global export. Importing it
  // before the host initializes can poison that module for all extensions.
  await page.waitForFunction(()=>!!window.comfyAPI?.app?.app,null,{timeout:60000});
  await page.waitForFunction(()=>{
   const app=window.comfyAPI.app.app;return !new URL(location.href).searchParams.has('stem_studio') && app.graph?._nodes?.length===6;
  },null,{timeout:75000});
  await page.waitForTimeout(1500);
  await page.screenshot({path:path.join(root,'stem-canvas-initial.png')});
  const graph=await page.evaluate(async()=>{const app=window.comfyAPI.app.app;return app.graph._nodes.map(n=>({type:n.type,widgets:n.widgets?.map(w=>({name:w.name,value:w.value}))}));});
  if(graph.length!==6)throw new Error('Expected six visible nodes; got '+JSON.stringify(graph));
  if(!graph.find(n=>n.type==='SSLoadSong').widgets.some(w=>w.name==='Upload song'))throw new Error('Upload button missing');
  const upload=await browser.newPage();
  await upload.goto(base+'/stem/upload');
  for(const name of ['large-synthetic.wav','synthetic.wav']){
   await upload.locator('#song').setInputFiles(path.join(root,name));
   await upload.locator('#send').click();
   await upload.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Upload complete:'),null,{timeout:60000});
   console.log('Browser upload complete:',name,fs.statSync(path.join(root,name)).size);
  }
  await page.waitForFunction(()=>{const app=window.comfyAPI.app.app;const n=app.graph._nodes.find(n=>n.type==='SSLoadSong');return n.widgets.find(w=>w.name==='audio').value.endsWith('_synthetic.wav');},null,{timeout:15000});
  // Prove the ordinary canvas serializes a backend prompt with matching names.
  const prompt=await page.evaluate(async()=>{const app=window.comfyAPI.app.app;return app.graphToPrompt();});
  if(!prompt.output['1'].inputs.audio.endsWith('_synthetic.wav'))throw new Error('Uploaded song selection did not reach the Run prompt');
  fs.writeFileSync(path.join(root,'browser-prompt.json'),JSON.stringify(prompt,null,2));
  const info=await(await page.request.get(base+'/object_info')).json();
  for(const node of Object.values(prompt.output))if(!info[node.class_type])throw new Error('Unregistered node: '+node.class_type);
  if(process.env.STEM_TEST_QUEUE==='1'){
   await page.evaluate(async()=>{const app=window.comfyAPI.app.app;await app.queuePrompt(0,1);});
   await page.waitForFunction(()=>{const app=window.comfyAPI.app.app;return app.graph?._nodes?.find(n=>n.type==='SSExport')?.widgets?.some(w=>w.name==='Download all stems ZIP');},null,{timeout:30000});
   const job=await page.evaluate(async()=>{const app=window.comfyAPI.app.app;return app.graph._nodes.find(n=>n.type==='SSExport')._stemJob;});
   const response=await page.request.get(base+'/stem/download/'+job);
   if(!response.ok())throw new Error('ZIP download failed');
   fs.writeFileSync(path.join(root,'browser-downloaded-stems.zip'),await response.body());
   console.log('Canvas Run, preview/export completion and ZIP download passed.');
  }
  await page.keyboard.press('Escape');
  await page.screenshot({path:path.join(root,'stem-canvas.png'),fullPage:false});
  console.log('Graph serialized:',Object.keys(prompt.output).length,'nodes. Page errors:',JSON.stringify(errors));
  if(errors.length)throw new Error('Browser errors');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
