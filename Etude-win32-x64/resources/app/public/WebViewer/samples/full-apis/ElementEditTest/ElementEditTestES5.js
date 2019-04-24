!function(e){"use strict";var f,s=Object.prototype.hasOwnProperty,t="function"==typeof Symbol?Symbol:{},o=t.iterator||"@@iterator",r=t.toStringTag||"@@toStringTag",n="object"==typeof module,i=e.regeneratorRuntime;if(i)n&&(module.exports=i);else{(i=e.regeneratorRuntime=n?module.exports:{}).wrap=c;var p="suspendedStart",h="suspendedYield",y="executing",g="completed",d={},a=l.prototype=m.prototype;u.prototype=a.constructor=l,l.constructor=u,l[r]=u.displayName="GeneratorFunction",i.isGeneratorFunction=function(e){var t="function"==typeof e&&e.constructor;return!!t&&(t===u||"GeneratorFunction"===(t.displayName||t.name))},i.mark=function(e){return Object.setPrototypeOf?Object.setPrototypeOf(e,l):(e.__proto__=l,r in e||(e[r]="GeneratorFunction")),e.prototype=Object.create(a),e},i.awrap=function(e){return new w(e)},x(b.prototype),i.async=function(e,t,r,n){var o=new b(c(e,t,r,n));return i.isGeneratorFunction(t)?o:o.next().then(function(e){return e.done?e.value:o.next()})},x(a),a[o]=function(){return this},a[r]="Generator",a.toString=function(){return"[object Generator]"},i.keys=function(r){var n=[];for(var e in r)n.push(e);return n.reverse(),function e(){for(;n.length;){var t=n.pop();if(t in r)return e.value=t,e.done=!1,e}return e.done=!0,e}},i.values=L,F.prototype={constructor:F,reset:function(e){if(this.prev=0,this.next=0,this.sent=this._sent=f,this.done=!1,this.delegate=null,this.tryEntries.forEach(P),!e)for(var t in this)"t"===t.charAt(0)&&s.call(this,t)&&!isNaN(+t.slice(1))&&(this[t]=f)},stop:function(){this.done=!0;var e=this.tryEntries[0].completion;if("throw"===e.type)throw e.arg;return this.rval},dispatchException:function(r){if(this.done)throw r;var n=this;function e(e,t){return i.type="throw",i.arg=r,n.next=e,!!t}for(var t=this.tryEntries.length-1;0<=t;--t){var o=this.tryEntries[t],i=o.completion;if("root"===o.tryLoc)return e("end");if(o.tryLoc<=this.prev){var a=s.call(o,"catchLoc"),c=s.call(o,"finallyLoc");if(a&&c){if(this.prev<o.catchLoc)return e(o.catchLoc,!0);if(this.prev<o.finallyLoc)return e(o.finallyLoc)}else if(a){if(this.prev<o.catchLoc)return e(o.catchLoc,!0)}else{if(!c)throw new Error("try statement without catch or finally");if(this.prev<o.finallyLoc)return e(o.finallyLoc)}}}},abrupt:function(e,t){for(var r=this.tryEntries.length-1;0<=r;--r){var n=this.tryEntries[r];if(n.tryLoc<=this.prev&&s.call(n,"finallyLoc")&&this.prev<n.finallyLoc){var o=n;break}}o&&("break"===e||"continue"===e)&&o.tryLoc<=t&&t<=o.finallyLoc&&(o=null);var i=o?o.completion:{};return i.type=e,i.arg=t,o?this.next=o.finallyLoc:this.complete(i),d},complete:function(e,t){if("throw"===e.type)throw e.arg;"break"===e.type||"continue"===e.type?this.next=e.arg:"return"===e.type?(this.rval=e.arg,this.next="end"):"normal"===e.type&&t&&(this.next=t)},finish:function(e){for(var t=this.tryEntries.length-1;0<=t;--t){var r=this.tryEntries[t];if(r.finallyLoc===e)return this.complete(r.completion,r.afterLoc),P(r),d}},catch:function(e){for(var t=this.tryEntries.length-1;0<=t;--t){var r=this.tryEntries[t];if(r.tryLoc===e){var n=r.completion;if("throw"===n.type){var o=n.arg;P(r)}return o}}throw new Error("illegal catch attempt")},delegateYield:function(e,t,r){return this.delegate={iterator:L(e),resultName:t,nextLoc:r},d}}}function c(e,t,r,n){var a,c,s,u,o=t&&t.prototype instanceof m?t:m,i=Object.create(o.prototype),l=new F(n||[]);return i._invoke=(a=e,c=r,s=l,u=p,function(e,t){if(u===y)throw new Error("Generator is already running");if(u===g){if("throw"===e)throw t;return D()}for(;;){var r=s.delegate;if(r){if("return"===e||"throw"===e&&r.iterator[e]===f){s.delegate=null;var n=r.iterator.return;if(n){var o=v(n,r.iterator,t);if("throw"===o.type){e="throw",t=o.arg;continue}}if("return"===e)continue}var o=v(r.iterator[e],r.iterator,t);if("throw"===o.type){s.delegate=null,e="throw",t=o.arg;continue}e="next",t=f;var i=o.arg;if(!i.done)return u=h,i;s[r.resultName]=i.value,s.next=r.nextLoc,s.delegate=null}if("next"===e)s.sent=s._sent=t;else if("throw"===e){if(u===p)throw u=g,t;s.dispatchException(t)&&(e="next",t=f)}else"return"===e&&s.abrupt("return",t);u=y;var o=v(a,c,s);if("normal"===o.type){u=s.done?g:h;var i={value:o.arg,done:s.done};if(o.arg!==d)return i;s.delegate&&"next"===e&&(t=f)}else"throw"===o.type&&(u=g,e="throw",t=o.arg)}}),i}function v(e,t,r){try{return{type:"normal",arg:e.call(t,r)}}catch(e){return{type:"throw",arg:e}}}function m(){}function u(){}function l(){}function x(e){["next","throw","return"].forEach(function(t){e[t]=function(e){return this._invoke(t,e)}})}function w(e){this.arg=e}function b(c){function s(e,t,r,n){var o=v(c[e],c,t);if("throw"!==o.type){var i=o.arg,a=i.value;return a instanceof w?Promise.resolve(a.arg).then(function(e){s("next",e,r,n)},function(e){s("throw",e,r,n)}):Promise.resolve(a).then(function(e){i.value=e,r(i)},n)}n(o.arg)}var t;"object"==typeof process&&process.domain&&(s=process.domain.bind(s)),this._invoke=function(r,n){function e(){return new Promise(function(e,t){s(r,n,e,t)})}return t=t?t.then(e,e):e()}}function E(e){var t={tryLoc:e[0]};1 in e&&(t.catchLoc=e[1]),2 in e&&(t.finallyLoc=e[2],t.afterLoc=e[3]),this.tryEntries.push(t)}function P(e){var t=e.completion||{};t.type="normal",delete t.arg,e.completion=t}function F(e){this.tryEntries=[{tryLoc:"root"}],e.forEach(E,this),this.reset(!0)}function L(t){if(t){var e=t[o];if(e)return e.call(t);if("function"==typeof t.next)return t;if(!isNaN(t.length)){var r=-1,n=function e(){for(;++r<t.length;)if(s.call(t,r))return e.value=t[r],e.done=!1,e;return e.value=f,e.done=!0,e};return n.next=n}}return{next:D}}function D(){return{value:f,done:!0}}}("object"==typeof global?global:"object"==typeof window?window:"object"==typeof self?self:this),function(e){"use strict";window.runElementEditTest=function(){var e=[y,t].map(regeneratorRuntime.mark);function y(t,r,n){var o,i,a,c,s,u,l,f,p,h;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,PDFNet.startDeallocateStack();case 2:return e.next=4,PDFNet.ColorSpace.createDeviceRGB();case 4:return i=e.sent,e.next=7,PDFNet.ColorPt.init(1,0,0,0);case 7:return a=e.sent,e.next=10,PDFNet.ColorPt.init(0,0,1,0);case 10:return c=e.sent,e.next=13,t.next();case 13:o=e.sent;case 14:if(null!==o)return e.next=17,o.getType();e.next=61;break;case 17:s=e.sent,e.t0=s,e.next=e.t0===PDFNet.Element.Type.e_image?21:e.t0===PDFNet.Element.Type.e_inline_image?21:e.t0===PDFNet.Element.Type.e_path?22:e.t0===PDFNet.Element.Type.e_text?29:e.t0===PDFNet.Element.Type.e_form?36:55;break;case 21:return e.abrupt("break",56);case 22:return e.next=24,o.getGState();case 24:return(u=e.sent).setFillColorSpace(i),u.setFillColorWithColorPt(a),r.writeElement(o),e.abrupt("break",56);case 29:return e.next=31,o.getGState();case 31:return(u=e.sent).setFillColorSpace(i),u.setFillColorWithColorPt(c),r.writeElement(o),e.abrupt("break",56);case 36:return r.writeElement(o),e.next=39,o.getXObject();case 39:if(l=e.sent,f=l.getObjNum(),-1===n.indexOf(f))return e.next=44,l.getObjNum();e.next=54;break;case 44:return p=e.sent,null==_.findWhere(n,p)&&n.push(p),e.next=48,PDFNet.ElementWriter.create();case 48:return h=e.sent,t.formBegin(),h.beginOnObj(l,!0),e.delegateYield(y(t,h,n),"t1",52);case 52:h.end(),t.end();case 54:return e.abrupt("break",56);case 55:r.writeElement(o);case 56:return e.next=58,t.next();case 58:o=e.sent,e.next=14;break;case 61:return e.next=63,PDFNet.endDeallocateStack();case 63:case"end":return e.stop()}},e[0],this)}function t(){var t,r,n,o,i,a,c,s,u,l,f,p;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return console.log("Beginning Test"),t=0,"../TestFiles/",e.next=5,PDFNet.PDFDoc.createFromURL("../TestFiles/newsletter.pdf");case 5:return(r=e.sent).initSecurityHandler(),r.lock(),console.log("PDF document initialized and locked"),e.next=11,PDFNet.ElementWriter.create();case 11:return n=e.sent,e.next=14,PDFNet.ElementReader.create();case 14:return o=e.sent,i=[],e.next=18,r.getPageCount();case 18:return a=e.sent,e.next=21,r.getPageIterator(1);case 21:c=e.sent;case 23:return e.next=25,c.hasNext();case 25:if(e.sent)return e.next=28,c.current();e.next=48;break;case 28:return s=e.sent,e.next=31,s.getIndex();case 31:return u=e.sent,console.log("Processing elements on page "+u+"/"+a),e.next=35,s.getSDFObj();case 35:return l=e.sent,e.next=38,l.getObjNum();case 38:return f=e.sent,null==_.findWhere(i,f)&&i.push(f),o.beginOnPage(s),n.beginOnPage(s,PDFNet.ElementWriter.WriteMode.e_replacement,!1),e.delegateYield(y(o,n,i),"t0",43);case 43:n.end(),o.end();case 45:c.next(),e.next=23;break;case 48:return e.next=50,r.saveMemoryBuffer(PDFNet.SDFDoc.SaveOptions.e_remove_unused);case 50:return p=e.sent,saveBufferAsPDFDoc(p,"newsletter_edited.pdf"),console.log("Done."),e.abrupt("return",t);case 54:case"end":return e.stop()}},e[1],this)}PDFNet.runGeneratorWithCleanup(t(),window.sampleL)}}();