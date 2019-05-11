!function(e){"use strict";var x,i=Object.prototype.hasOwnProperty,t="function"==typeof Symbol?Symbol:{},o=t.iterator||"@@iterator",n=t.toStringTag||"@@toStringTag",r="object"==typeof module,a=e.regeneratorRuntime;if(a)r&&(module.exports=a);else{(a=e.regeneratorRuntime=r?module.exports:{}).wrap=c;var f="suspendedStart",g="suspendedYield",p="executing",h="completed",d={},s=l.prototype=b.prototype;u.prototype=s.constructor=l,l.constructor=u,l[n]=u.displayName="GeneratorFunction",a.isGeneratorFunction=function(e){var t="function"==typeof e&&e.constructor;return!!t&&(t===u||"GeneratorFunction"===(t.displayName||t.name))},a.mark=function(e){return Object.setPrototypeOf?Object.setPrototypeOf(e,l):(e.__proto__=l,n in e||(e[n]="GeneratorFunction")),e.prototype=Object.create(s),e},a.awrap=function(e){return new m(e)},v(w.prototype),a.async=function(e,t,n,r){var o=new w(c(e,t,n,r));return a.isGeneratorFunction(t)?o:o.next().then(function(e){return e.done?e.value:o.next()})},v(s),s[o]=function(){return this},s[n]="Generator",s.toString=function(){return"[object Generator]"},a.keys=function(n){var r=[];for(var e in n)r.push(e);return r.reverse(),function e(){for(;r.length;){var t=r.pop();if(t in n)return e.value=t,e.done=!1,e}return e.done=!0,e}},a.values=N,P.prototype={constructor:P,reset:function(e){if(this.prev=0,this.next=0,this.sent=this._sent=x,this.done=!1,this.delegate=null,this.tryEntries.forEach(F),!e)for(var t in this)"t"===t.charAt(0)&&i.call(this,t)&&!isNaN(+t.slice(1))&&(this[t]=x)},stop:function(){this.done=!0;var e=this.tryEntries[0].completion;if("throw"===e.type)throw e.arg;return this.rval},dispatchException:function(n){if(this.done)throw n;var r=this;function e(e,t){return a.type="throw",a.arg=n,r.next=e,!!t}for(var t=this.tryEntries.length-1;0<=t;--t){var o=this.tryEntries[t],a=o.completion;if("root"===o.tryLoc)return e("end");if(o.tryLoc<=this.prev){var s=i.call(o,"catchLoc"),c=i.call(o,"finallyLoc");if(s&&c){if(this.prev<o.catchLoc)return e(o.catchLoc,!0);if(this.prev<o.finallyLoc)return e(o.finallyLoc)}else if(s){if(this.prev<o.catchLoc)return e(o.catchLoc,!0)}else{if(!c)throw new Error("try statement without catch or finally");if(this.prev<o.finallyLoc)return e(o.finallyLoc)}}}},abrupt:function(e,t){for(var n=this.tryEntries.length-1;0<=n;--n){var r=this.tryEntries[n];if(r.tryLoc<=this.prev&&i.call(r,"finallyLoc")&&this.prev<r.finallyLoc){var o=r;break}}o&&("break"===e||"continue"===e)&&o.tryLoc<=t&&t<=o.finallyLoc&&(o=null);var a=o?o.completion:{};return a.type=e,a.arg=t,o?this.next=o.finallyLoc:this.complete(a),d},complete:function(e,t){if("throw"===e.type)throw e.arg;"break"===e.type||"continue"===e.type?this.next=e.arg:"return"===e.type?(this.rval=e.arg,this.next="end"):"normal"===e.type&&t&&(this.next=t)},finish:function(e){for(var t=this.tryEntries.length-1;0<=t;--t){var n=this.tryEntries[t];if(n.finallyLoc===e)return this.complete(n.completion,n.afterLoc),F(n),d}},catch:function(e){for(var t=this.tryEntries.length-1;0<=t;--t){var n=this.tryEntries[t];if(n.tryLoc===e){var r=n.completion;if("throw"===r.type){var o=r.arg;F(n)}return o}}throw new Error("illegal catch attempt")},delegateYield:function(e,t,n){return this.delegate={iterator:N(e),resultName:t,nextLoc:n},d}}}function c(e,t,n,r){var s,c,i,u,o=t&&t.prototype instanceof b?t:b,a=Object.create(o.prototype),l=new P(r||[]);return a._invoke=(s=e,c=n,i=l,u=f,function(e,t){if(u===p)throw new Error("Generator is already running");if(u===h){if("throw"===e)throw t;return L()}for(;;){var n=i.delegate;if(n){if("return"===e||"throw"===e&&n.iterator[e]===x){i.delegate=null;var r=n.iterator.return;if(r){var o=y(r,n.iterator,t);if("throw"===o.type){e="throw",t=o.arg;continue}}if("return"===e)continue}var o=y(n.iterator[e],n.iterator,t);if("throw"===o.type){i.delegate=null,e="throw",t=o.arg;continue}e="next",t=x;var a=o.arg;if(!a.done)return u=g,a;i[n.resultName]=a.value,i.next=n.nextLoc,i.delegate=null}if("next"===e)i.sent=i._sent=t;else if("throw"===e){if(u===f)throw u=h,t;i.dispatchException(t)&&(e="next",t=x)}else"return"===e&&i.abrupt("return",t);u=p;var o=y(s,c,i);if("normal"===o.type){u=i.done?h:g;var a={value:o.arg,done:i.done};if(o.arg!==d)return a;i.delegate&&"next"===e&&(t=x)}else"throw"===o.type&&(u=h,e="throw",t=o.arg)}}),a}function y(e,t,n){try{return{type:"normal",arg:e.call(t,n)}}catch(e){return{type:"throw",arg:e}}}function b(){}function u(){}function l(){}function v(e){["next","throw","return"].forEach(function(t){e[t]=function(e){return this._invoke(t,e)}})}function m(e){this.arg=e}function w(c){function i(e,t,n,r){var o=y(c[e],c,t);if("throw"!==o.type){var a=o.arg,s=a.value;return s instanceof m?Promise.resolve(s.arg).then(function(e){i("next",e,n,r)},function(e){i("throw",e,n,r)}):Promise.resolve(s).then(function(e){a.value=e,n(a)},r)}r(o.arg)}var t;"object"==typeof process&&process.domain&&(i=process.domain.bind(i)),this._invoke=function(n,r){function e(){return new Promise(function(e,t){i(n,r,e,t)})}return t=t?t.then(e,e):e()}}function k(e){var t={tryLoc:e[0]};1 in e&&(t.catchLoc=e[1]),2 in e&&(t.finallyLoc=e[2],t.afterLoc=e[3]),this.tryEntries.push(t)}function F(e){var t=e.completion||{};t.type="normal",delete t.arg,e.completion=t}function P(e){this.tryEntries=[{tryLoc:"root"}],e.forEach(k,this),this.reset(!0)}function N(t){if(t){var e=t[o];if(e)return e.call(t);if("function"==typeof t.next)return t;if(!isNaN(t.length)){var n=-1,r=function e(){for(;++n<t.length;)if(i.call(t,n))return e.value=t[n],e.done=!1,e;return e.value=x,e.done=!0,e};return r.next=r}}return{next:L}}function L(){return{value:x,done:!0}}}("object"==typeof global?global:"object"==typeof window?window:"object"==typeof self?self:this),function(e){"use strict";window.runTextExtractTest=function(){var e=[T,c,S,B,t].map(regeneratorRuntime.mark);function T(t){var n,r,o;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,t.next();case 2:if(e.t0=n=e.sent,null!==e.t0)return e.next=6,n.getType();e.next=29;break;case 6:e.t1=e.sent,e.next=e.t1===PDFNet.Element.Type.e_text_begin?9:e.t1===PDFNet.Element.Type.e_text_end?11:e.t1===PDFNet.Element.Type.e_text?13:e.t1===PDFNet.Element.Type.e_text_new_line?22:e.t1===PDFNet.Element.Type.e_form?23:27;break;case 9:return console.log("--\x3e Text Block Begin"),e.abrupt("break",27);case 11:return console.log("--\x3e Text Block End"),e.abrupt("break",27);case 13:return e.next=15,n.getBBox();case 15:return r=e.sent,console.log("--\x3e BBox: "+r.x1+", "+r.y1+", "+r.x2+", "+r.y2+"\n"),e.next=19,n.getTextString();case 19:return o=e.sent,console.log(o),e.abrupt("break",27);case 22:return e.abrupt("break",27);case 23:return t.formBegin(),e.delegateYield(T(t),"t2",25);case 25:return t.end(),e.abrupt("break",27);case 27:e.next=0;break;case 29:case"end":return e.stop()}},e[0],this)}function c(t,n,r){var o,a,s;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,t.next();case 2:if(e.t0=o=e.sent,null!==e.t0)return e.next=6,o.getType();e.next=28;break;case 6:e.t1=e.sent,e.next=e.t1===PDFNet.Element.Type.e_text?9:e.t1===PDFNet.Element.Type.e_text_new_line?20:e.t1===PDFNet.Element.Type.e_form?21:26;break;case 9:return e.next=11,o.getBBox();case 11:return a=e.sent,e.next=14,a.intersectRect(a,n);case 14:if(e.sent)return e.next=17,o.getTextString();e.next=19;break;case 17:s=e.sent,r+=s+"\n";case 19:case 20:return e.abrupt("break",26);case 21:return t.formBegin(),e.delegateYield(c(t,n,r),"t2",23);case 23:return r+=e.t2,t.end(),e.abrupt("break",26);case 26:e.next=0;break;case 28:return e.abrupt("return",r);case 29:case"end":return e.stop()}},e[1],this)}function S(t,n,r){var o;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return o="",r.beginOnPage(t),e.delegateYield(c(r,n,o),"t0",3);case 3:return o+=e.t0,r.end(),e.abrupt("return",o);case 6:case"end":return e.stop()}},e[2],this)}function B(t){var n,r,o,a,s,c,i,u;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,t.getColor();case 2:return n=e.sent,e.next=5,n.get(0);case 5:return r=e.sent,e.next=8,n.get(1);case 8:return o=e.sent,e.next=11,n.get(2);case 11:return a=e.sent,e.next=14,t.getFontName();case 14:return s=e.sent,e.next=17,t.getFontSize();case 17:return c=e.sent,e.next=20,t.isSerif();case 20:if(!e.sent){e.next=24;break}e.t0=" sans-serif; ",e.next=25;break;case 24:e.t0=" ";case 25:return i=e.t0,u='style="font-family:'+s+";font-size:"+c+";"+i+"color: #"+r.toString(16)+", "+o.toString(16)+", "+a.toString(16)+')"',e.abrupt("return",u);case 28:case"end":return e.stop()}},e[3],this)}function t(){var t,n,r,o,a,s,c,i,u,l,x,f,g,p,h,d,y,b,v,m,w,k,F,P,N,L,D,E,_;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return console.log("Beginning Test"),0,e.next=4,PDFNet.initialize();case 4:return t="../TestFiles/",c=!(s=!(a=o=r=!(n="newsletter.pdf"))),e.prev=11,e.next=14,PDFNet.startDeallocateStack();case 14:return e.next=16,PDFNet.PDFDoc.createFromURL(t+n);case 16:return(i=e.sent).initSecurityHandler(),i.lock(),e.next=21,i.getPage(1);case 21:if("0"===(u=e.sent).id)return console.log("Page not found."),e.abrupt("return",1);e.next=25;break;case 25:return e.next=27,PDFNet.TextExtractor.create();case 27:return l=e.sent,x=new PDFNet.Rect(0,0,612,794),l.begin(u,x),e.next=32,l.getNumLines();case 32:if(e.sent,r)return e.next=36,l.getWordCount();e.next=44;break;case 36:return h=e.sent,console.log("Word Count: "+h),e.next=40,l.getAsText();case 40:f=e.sent,console.log("- GetAsText  -------------------------------"),console.log(f),console.log("-----------------------------------------");case 44:if(o)return e.next=47,l.getAsXML(PDFNet.TextExtractor.XMLOutputFlags.e_words_as_elements|PDFNet.TextExtractor.XMLOutputFlags.e_output_bbox|PDFNet.TextExtractor.XMLOutputFlags.e_output_style_info);e.next=50;break;case 47:f=e.sent,console.log("- GetAsXML  --------------------------"+f),console.log("-----------------------------------------------------------");case 50:if(a)return e.next=53,l.getFirstLine();e.next=78;break;case 53:g=e.sent;case 54:return e.next=56,g.isValid();case 56:if(e.sent)return e.next=59,g.getFirstWord();e.next=77;break;case 59:p=e.sent;case 60:return e.next=62,p.isValid();case 62:if(e.sent)return e.next=65,p.getString();e.next=72;break;case 65:f=e.sent,console.log(f);case 67:return e.next=69,p.getNextWord();case 69:p=e.sent,e.next=60;break;case 72:return e.next=74,g.getNextLine();case 74:g=e.sent,e.next=54;break;case 77:console.log("-----------------------------------------------------------");case 78:if(s)return v=b=-1,e.next=83,PDFNet.ElementBuilder.create();e.next=179;break;case 83:return e.sent,e.next=86,PDFNet.ElementWriter.create();case 86:return e.sent,e.next=89,l.getFirstLine();case 89:g=e.sent;case 90:return e.next=92,g.isValid();case 92:if(e.sent)return e.next=95,g.getNumWords();e.next=178;break;case 95:if(e.t0=e.sent,0===e.t0)return e.abrupt("continue",173);e.next=98;break;case 98:return e.next=100,g.getFlowID();case 100:if(e.t1=e.sent,e.t2=b,e.t1!==e.t2)return-1!==b&&(-1!==v&&(v=-1,console.log("</Para>")),console.log("</Flow>")),e.next=106,g.getFlowID();e.next=108;break;case 106:b=e.sent,console.log('<Flow id="'+b+'">');case 108:return e.next=110,g.getParagraphID();case 110:if(e.t3=e.sent,e.t4=v,e.t3!==e.t4)return-1!==v&&console.log("</Para>"),e.next=116,g.getParagraphID();e.next=118;break;case 116:v=e.sent,console.log('<Para id="'+v+'">');case 118:return e.next=120,g.getBBox();case 120:return d=e.sent,e.next=123,g.getStyle();case 123:return m=e.sent,w='<Line box="'+d.x1+", "+d.y1+", "+d.x2+", "+d.y1+'">',e.delegateYield(B(m),"t5",126);case 126:return w+=e.t5,e.next=129,g.getCurrentNum();case 129:return k=e.sent,w+=' cur_num="'+k+'">',console.log(w),F="",e.next=135,g.getFirstWord();case 135:p=e.sent;case 136:return e.next=138,p.isValid();case 138:if(e.sent)return e.next=141,p.getBBox();e.next=172;break;case 141:return y=e.sent,e.next=144,p.getCurrentNum();case 144:return P=e.sent,F+='<Word box="'+y.x1+", "+y.y1+", "+y.x2+", "+y.y2+'" cur_num="'+P+'"',e.next=148,p.getStringLen();case 148:if(0===e.sent)return e.abrupt("continue",167);e.next=151;break;case 151:return e.next=153,p.getStyle();case 153:return N=e.sent,e.next=156,N.compare(m);case 156:if(e.sent){e.next=161;break}return e.t6=console,e.delegateYield(B(N),"t7",159);case 159:e.t8=e.t7,e.t6.log.call(e.t6,e.t8);case 161:return e.next=163,p.getString();case 163:e.t9=e.sent,e.t10=">"+e.t9,F+=e.t10+"</Word>",console.log(F);case 167:return e.next=169,p.getNextWord();case 169:p=e.sent,e.next=136;break;case 172:console.log("</Line>");case 173:return e.next=175,g.getNextLine();case 175:g=e.sent,e.next=90;break;case 178:-1!==b&&(-1!==v&&(v=-1,console.log("</Para>")),console.log("</Flow>\n"));case 179:return console.log("done"),e.next=182,PDFNet.endDeallocateStack();case 182:e.next=189;break;case 184:e.prev=184,e.t11=e.catch(11),console.log(e.t11),console.log(e.t11.stack),1;case 189:if(c)return 0,e.prev=191,e.next=194,PDFNet.startDeallocateStack();e.next=259;break;case 194:return e.next=196,PDFNet.PDFDoc.createFromURL(t+n);case 196:return(i=e.sent).initSecurityHandler(),i.lock(),e.next=201,PDFNet.ElementReader.create();case 201:return L=e.sent,e.next=204,i.getPageIterator(1);case 204:D=e.sent;case 206:return e.next=208,D.hasNext();case 208:if(e.sent)return e.next=211,D.current();e.next=218;break;case 211:return u=e.sent,L.beginOnPage(u),e.delegateYield(T(L),"t12",214);case 214:L.end();case 215:D.next(),e.next=206;break;case 218:return console.log("----------------------------------------------------"),console.log("Extract text based on the selection rectangle."),console.log("----------------------------------------------------"),e.next=223,i.getPageIterator();case 223:return e.next=225,e.sent.current();case 225:return E=e.sent,e.t13=E,e.next=229,PDFNet.Rect.init(27,392,563,534);case 229:return e.t14=e.sent,e.t15=L,e.delegateYield(S(e.t13,e.t14,e.t15),"t16",232);case 232:return _=e.t16,console.log("Field 1: "+_),e.t17=E,e.next=237,PDFNet.Rect.init(28,551,106,623);case 237:return e.t18=e.sent,e.t19=L,e.delegateYield(S(e.t17,e.t18,e.t19),"t20",240);case 240:return _=e.t20,console.log("Field 2: "+_),e.t21=E,e.next=245,PDFNet.Rect.init(208,550,387,621);case 245:return e.t22=e.sent,e.t23=L,e.delegateYield(S(e.t21,e.t22,e.t23),"t24",248);case 248:return _=e.t24,console.log("Field 3: "+_),console.log("Done"),e.next=253,PDFNet.endDeallocateStack();case 253:e.next=259;break;case 255:e.prev=255,e.t25=e.catch(191),console.log(e.t25.stack),1;case 259:case"end":return e.stop()}},e[4],this,[[11,184],[191,255]])}PDFNet.runGeneratorWithCleanup(t(),window.sampleL)}}();