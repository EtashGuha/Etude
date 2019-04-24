!function(e){"use strict";var f,s=Object.prototype.hasOwnProperty,t="function"==typeof Symbol?Symbol:{},o=t.iterator||"@@iterator",r=t.toStringTag||"@@toStringTag",n="object"==typeof module,i=e.regeneratorRuntime;if(i)n&&(module.exports=i);else{(i=e.regeneratorRuntime=n?module.exports:{}).wrap=c;var p="suspendedStart",h="suspendedYield",d="executing",v="completed",y={},a=l.prototype=x.prototype;u.prototype=a.constructor=l,l.constructor=u,l[r]=u.displayName="GeneratorFunction",i.isGeneratorFunction=function(e){var t="function"==typeof e&&e.constructor;return!!t&&(t===u||"GeneratorFunction"===(t.displayName||t.name))},i.mark=function(e){return Object.setPrototypeOf?Object.setPrototypeOf(e,l):(e.__proto__=l,r in e||(e[r]="GeneratorFunction")),e.prototype=Object.create(a),e},i.awrap=function(e){return new w(e)},m(L.prototype),i.async=function(e,t,r,n){var o=new L(c(e,t,r,n));return i.isGeneratorFunction(t)?o:o.next().then(function(e){return e.done?e.value:o.next()})},m(a),a[o]=function(){return this},a[r]="Generator",a.toString=function(){return"[object Generator]"},i.keys=function(r){var n=[];for(var e in r)n.push(e);return n.reverse(),function e(){for(;n.length;){var t=n.pop();if(t in r)return e.value=t,e.done=!1,e}return e.done=!0,e}},i.values=S,E.prototype={constructor:E,reset:function(e){if(this.prev=0,this.next=0,this.sent=this._sent=f,this.done=!1,this.delegate=null,this.tryEntries.forEach(F),!e)for(var t in this)"t"===t.charAt(0)&&s.call(this,t)&&!isNaN(+t.slice(1))&&(this[t]=f)},stop:function(){this.done=!0;var e=this.tryEntries[0].completion;if("throw"===e.type)throw e.arg;return this.rval},dispatchException:function(r){if(this.done)throw r;var n=this;function e(e,t){return i.type="throw",i.arg=r,n.next=e,!!t}for(var t=this.tryEntries.length-1;0<=t;--t){var o=this.tryEntries[t],i=o.completion;if("root"===o.tryLoc)return e("end");if(o.tryLoc<=this.prev){var a=s.call(o,"catchLoc"),c=s.call(o,"finallyLoc");if(a&&c){if(this.prev<o.catchLoc)return e(o.catchLoc,!0);if(this.prev<o.finallyLoc)return e(o.finallyLoc)}else if(a){if(this.prev<o.catchLoc)return e(o.catchLoc,!0)}else{if(!c)throw new Error("try statement without catch or finally");if(this.prev<o.finallyLoc)return e(o.finallyLoc)}}}},abrupt:function(e,t){for(var r=this.tryEntries.length-1;0<=r;--r){var n=this.tryEntries[r];if(n.tryLoc<=this.prev&&s.call(n,"finallyLoc")&&this.prev<n.finallyLoc){var o=n;break}}o&&("break"===e||"continue"===e)&&o.tryLoc<=t&&t<=o.finallyLoc&&(o=null);var i=o?o.completion:{};return i.type=e,i.arg=t,o?this.next=o.finallyLoc:this.complete(i),y},complete:function(e,t){if("throw"===e.type)throw e.arg;"break"===e.type||"continue"===e.type?this.next=e.arg:"return"===e.type?(this.rval=e.arg,this.next="end"):"normal"===e.type&&t&&(this.next=t)},finish:function(e){for(var t=this.tryEntries.length-1;0<=t;--t){var r=this.tryEntries[t];if(r.finallyLoc===e)return this.complete(r.completion,r.afterLoc),F(r),y}},catch:function(e){for(var t=this.tryEntries.length-1;0<=t;--t){var r=this.tryEntries[t];if(r.tryLoc===e){var n=r.completion;if("throw"===n.type){var o=n.arg;F(r)}return o}}throw new Error("illegal catch attempt")},delegateYield:function(e,t,r){return this.delegate={iterator:S(e),resultName:t,nextLoc:r},y}}}function c(e,t,r,n){var a,c,s,u,o=t&&t.prototype instanceof x?t:x,i=Object.create(o.prototype),l=new E(n||[]);return i._invoke=(a=e,c=r,s=l,u=p,function(e,t){if(u===d)throw new Error("Generator is already running");if(u===v){if("throw"===e)throw t;return P()}for(;;){var r=s.delegate;if(r){if("return"===e||"throw"===e&&r.iterator[e]===f){s.delegate=null;var n=r.iterator.return;if(n){var o=g(n,r.iterator,t);if("throw"===o.type){e="throw",t=o.arg;continue}}if("return"===e)continue}var o=g(r.iterator[e],r.iterator,t);if("throw"===o.type){s.delegate=null,e="throw",t=o.arg;continue}e="next",t=f;var i=o.arg;if(!i.done)return u=h,i;s[r.resultName]=i.value,s.next=r.nextLoc,s.delegate=null}if("next"===e)s.sent=s._sent=t;else if("throw"===e){if(u===p)throw u=v,t;s.dispatchException(t)&&(e="next",t=f)}else"return"===e&&s.abrupt("return",t);u=d;var o=g(a,c,s);if("normal"===o.type){u=s.done?v:h;var i={value:o.arg,done:s.done};if(o.arg!==y)return i;s.delegate&&"next"===e&&(t=f)}else"throw"===o.type&&(u=v,e="throw",t=o.arg)}}),i}function g(e,t,r){try{return{type:"normal",arg:e.call(t,r)}}catch(e){return{type:"throw",arg:e}}}function x(){}function u(){}function l(){}function m(e){["next","throw","return"].forEach(function(t){e[t]=function(e){return this._invoke(t,e)}})}function w(e){this.arg=e}function L(c){function s(e,t,r,n){var o=g(c[e],c,t);if("throw"!==o.type){var i=o.arg,a=i.value;return a instanceof w?Promise.resolve(a.arg).then(function(e){s("next",e,r,n)},function(e){s("throw",e,r,n)}):Promise.resolve(a).then(function(e){i.value=e,r(i)},n)}n(o.arg)}var t;"object"==typeof process&&process.domain&&(s=process.domain.bind(s)),this._invoke=function(r,n){function e(){return new Promise(function(e,t){s(r,n,e,t)})}return t=t?t.then(e,e):e()}}function D(e){var t={tryLoc:e[0]};1 in e&&(t.catchLoc=e[1]),2 in e&&(t.finallyLoc=e[2],t.afterLoc=e[3]),this.tryEntries.push(t)}function F(e){var t=e.completion||{};t.type="normal",delete t.arg,e.completion=t}function E(e){this.tryEntries=[{tryLoc:"root"}],e.forEach(D,this),this.reset(!0)}function S(t){if(t){var e=t[o];if(e)return e.call(t);if("function"==typeof t.next)return t;if(!isNaN(t.length)){var r=-1,n=function e(){for(;++r<t.length;)if(s.call(t,r))return e.value=t[r],e.done=!1,e;return e.value=f,e.done=!0,e};return n.next=n}}return{next:P}}function P(){return{value:f,done:!0}}}("object"==typeof global?global:"object"==typeof window?window:"object"==typeof self?self:this),function(e){"use strict";window.runContentReplacer=function(){var e=[t].map(regeneratorRuntime.mark);function t(){var t,r,n,o,i,a,c,s,u,l;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,console.log("Beginning Content Replacer Test"),t="../TestFiles/",r="BusinessCardTemplate.pdf",n="BusinessCard.pdf",e.next=7,PDFNet.initialize();case 7:return e.next=9,PDFNet.PDFDoc.createFromURL(t+r);case 9:return(o=e.sent).initSecurityHandler(),o.lock(),console.log("PDFNet and PDF document initialized and locked"),e.next=15,PDFNet.ContentReplacer.create();case 15:return i=e.sent,e.next=18,o.getPage(1);case 18:return a=e.sent,e.next=21,PDFNet.Image.createFromURL(o,t+"peppers.jpg");case 21:return c=e.sent,e.next=24,a.getMediaBox();case 24:return s=e.sent,e.next=27,c.getSDFObj();case 27:return u=e.sent,e.next=30,i.addImage(s,u);case 30:return e.next=32,i.addString("NAME","John Smith");case 32:return e.next=34,i.addString("QUALIFICATIONS","Philosophy Doctor");case 34:return e.next=36,i.addString("JOB_TITLE","Software Developer");case 36:return e.next=38,i.addString("ADDRESS_LINE1","#100 123 Software Rd");case 38:return e.next=40,i.addString("ADDRESS_LINE2","Vancouver, BC");case 40:return e.next=42,i.addString("PHONE_OFFICE","604-730-8989");case 42:return e.next=44,i.addString("PHONE_MOBILE","604-765-4321");case 44:return e.next=46,i.addString("EMAIL","info@pdftron.com");case 46:return e.next=48,i.addString("WEBSITE_URL","http://www.pdftron.com");case 48:return e.next=50,i.process(a);case 50:return e.next=52,o.saveMemoryBuffer(PDFNet.SDFDoc.SaveOptions.e_remove_unused);case 52:l=e.sent,saveBufferAsPDFDoc(l,n),console.log("Done. Result saved as "+n),e.next=60;break;case 57:e.prev=57,e.t0=e.catch(0),console.log(e.t0);case 60:return e.prev=60,console.log("Beginning Content Replacer Test"),t="../TestFiles/",r="newsletter.pdf",n="newsletterReplaced.pdf",e.next=67,PDFNet.initialize();case 67:return e.next=69,PDFNet.PDFDoc.createFromURL(t+r);case 69:return(o=e.sent).initSecurityHandler(),o.lock(),console.log("PDFNet and PDF document initialized and locked"),e.next=75,PDFNet.ContentReplacer.create();case 75:return i=e.sent,e.next=78,o.getPage(1);case 78:return a=e.sent,e.next=81,a.getMediaBox();case 81:return s=e.sent,e.next=84,i.addText(s,"The quick onyx goblin jumps over the lazy dwarf");case 84:return e.next=86,i.process(a);case 86:return e.next=88,o.saveMemoryBuffer(PDFNet.SDFDoc.SaveOptions.e_remove_unused);case 88:l=e.sent,saveBufferAsPDFDoc(l,n),console.log("Done. Result saved as "+n),e.next=96;break;case 93:e.prev=93,e.t1=e.catch(60),console.log(e.t1);case 96:case"end":return e.stop()}},e[0],this,[[0,57],[60,93]])}PDFNet.runGeneratorWithCleanup(t(),window.sampleL)}}();