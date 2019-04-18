!function(e){"use strict";var P,i=Object.prototype.hasOwnProperty,t="function"==typeof Symbol?Symbol:{},a=t.iterator||"@@iterator",n=t.toStringTag||"@@toStringTag",r="object"==typeof module,s=e.regeneratorRuntime;if(s)r&&(module.exports=s);else{(s=e.regeneratorRuntime=r?module.exports:{}).wrap=c;var x="suspendedStart",D="suspendedYield",p="executing",F="completed",N={},o=l.prototype=d.prototype;u.prototype=o.constructor=l,l.constructor=u,l[n]=u.displayName="GeneratorFunction",s.isGeneratorFunction=function(e){var t="function"==typeof e&&e.constructor;return!!t&&(t===u||"GeneratorFunction"===(t.displayName||t.name))},s.mark=function(e){return Object.setPrototypeOf?Object.setPrototypeOf(e,l):(e.__proto__=l,n in e||(e[n]="GeneratorFunction")),e.prototype=Object.create(o),e},s.awrap=function(e){return new g(e)},f(y.prototype),s.async=function(e,t,n,r){var a=new y(c(e,t,n,r));return s.isGeneratorFunction(t)?a:a.next().then(function(e){return e.done?e.value:a.next()})},f(o),o[a]=function(){return this},o[n]="Generator",o.toString=function(){return"[object Generator]"},s.keys=function(n){var r=[];for(var e in n)r.push(e);return r.reverse(),function e(){for(;r.length;){var t=r.pop();if(t in n)return e.value=t,e.done=!1,e}return e.done=!0,e}},s.values=C,S.prototype={constructor:S,reset:function(e){if(this.prev=0,this.next=0,this.sent=this._sent=P,this.done=!1,this.delegate=null,this.tryEntries.forEach(A),!e)for(var t in this)"t"===t.charAt(0)&&i.call(this,t)&&!isNaN(+t.slice(1))&&(this[t]=P)},stop:function(){this.done=!0;var e=this.tryEntries[0].completion;if("throw"===e.type)throw e.arg;return this.rval},dispatchException:function(n){if(this.done)throw n;var r=this;function e(e,t){return s.type="throw",s.arg=n,r.next=e,!!t}for(var t=this.tryEntries.length-1;0<=t;--t){var a=this.tryEntries[t],s=a.completion;if("root"===a.tryLoc)return e("end");if(a.tryLoc<=this.prev){var o=i.call(a,"catchLoc"),c=i.call(a,"finallyLoc");if(o&&c){if(this.prev<a.catchLoc)return e(a.catchLoc,!0);if(this.prev<a.finallyLoc)return e(a.finallyLoc)}else if(o){if(this.prev<a.catchLoc)return e(a.catchLoc,!0)}else{if(!c)throw new Error("try statement without catch or finally");if(this.prev<a.finallyLoc)return e(a.finallyLoc)}}}},abrupt:function(e,t){for(var n=this.tryEntries.length-1;0<=n;--n){var r=this.tryEntries[n];if(r.tryLoc<=this.prev&&i.call(r,"finallyLoc")&&this.prev<r.finallyLoc){var a=r;break}}a&&("break"===e||"continue"===e)&&a.tryLoc<=t&&t<=a.finallyLoc&&(a=null);var s=a?a.completion:{};return s.type=e,s.arg=t,a?this.next=a.finallyLoc:this.complete(s),N},complete:function(e,t){if("throw"===e.type)throw e.arg;"break"===e.type||"continue"===e.type?this.next=e.arg:"return"===e.type?(this.rval=e.arg,this.next="end"):"normal"===e.type&&t&&(this.next=t)},finish:function(e){for(var t=this.tryEntries.length-1;0<=t;--t){var n=this.tryEntries[t];if(n.finallyLoc===e)return this.complete(n.completion,n.afterLoc),A(n),N}},catch:function(e){for(var t=this.tryEntries.length-1;0<=t;--t){var n=this.tryEntries[t];if(n.tryLoc===e){var r=n.completion;if("throw"===r.type){var a=r.arg;A(n)}return a}}throw new Error("illegal catch attempt")},delegateYield:function(e,t,n){return this.delegate={iterator:C(e),resultName:t,nextLoc:n},N}}}function c(e,t,n,r){var o,c,i,u,a=t&&t.prototype instanceof d?t:d,s=Object.create(a.prototype),l=new S(r||[]);return s._invoke=(o=e,c=n,i=l,u=x,function(e,t){if(u===p)throw new Error("Generator is already running");if(u===F){if("throw"===e)throw t;return k()}for(;;){var n=i.delegate;if(n){if("return"===e||"throw"===e&&n.iterator[e]===P){i.delegate=null;var r=n.iterator.return;if(r){var a=h(r,n.iterator,t);if("throw"===a.type){e="throw",t=a.arg;continue}}if("return"===e)continue}var a=h(n.iterator[e],n.iterator,t);if("throw"===a.type){i.delegate=null,e="throw",t=a.arg;continue}e="next",t=P;var s=a.arg;if(!s.done)return u=D,s;i[n.resultName]=s.value,i.next=n.nextLoc,i.delegate=null}if("next"===e)i.sent=i._sent=t;else if("throw"===e){if(u===x)throw u=F,t;i.dispatchException(t)&&(e="next",t=P)}else"return"===e&&i.abrupt("return",t);u=p;var a=h(o,c,i);if("normal"===a.type){u=i.done?F:D;var s={value:a.arg,done:i.done};if(a.arg!==N)return s;i.delegate&&"next"===e&&(t=P)}else"throw"===a.type&&(u=F,e="throw",t=a.arg)}}),s}function h(e,t,n){try{return{type:"normal",arg:e.call(t,n)}}catch(e){return{type:"throw",arg:e}}}function d(){}function u(){}function l(){}function f(e){["next","throw","return"].forEach(function(t){e[t]=function(e){return this._invoke(t,e)}})}function g(e){this.arg=e}function y(c){function i(e,t,n,r){var a=h(c[e],c,t);if("throw"!==a.type){var s=a.arg,o=s.value;return o instanceof g?Promise.resolve(o.arg).then(function(e){i("next",e,n,r)},function(e){i("throw",e,n,r)}):Promise.resolve(o).then(function(e){s.value=e,n(s)},r)}r(a.arg)}var t;"object"==typeof process&&process.domain&&(i=process.domain.bind(i)),this._invoke=function(n,r){function e(){return new Promise(function(e,t){i(n,r,e,t)})}return t=t?t.then(e,e):e()}}function w(e){var t={tryLoc:e[0]};1 in e&&(t.catchLoc=e[1]),2 in e&&(t.finallyLoc=e[2],t.afterLoc=e[3]),this.tryEntries.push(t)}function A(e){var t=e.completion||{};t.type="normal",delete t.arg,e.completion=t}function S(e){this.tryEntries=[{tryLoc:"root"}],e.forEach(w,this),this.reset(!0)}function C(t){if(t){var e=t[a];if(e)return e.call(t);if("function"==typeof t.next)return t;if(!isNaN(t.length)){var n=-1,r=function e(){for(;++n<t.length;)if(i.call(t,n))return e.value=t[n],e.done=!1,e;return e.value=P,e.done=!0,e};return r.next=r}}return{next:k}}function k(){return{value:P,done:!0}}}("object"==typeof global?global:"object"==typeof window?window:"object"==typeof self?self:this),function(e){"use strict";window.runAnnotationTest=function(){var e=[c,i,u,t].map(regeneratorRuntime.mark);function c(t){var n,r,a,s,o,c,i,u,l,P,x,D,p;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,PDFNet.startDeallocateStack();case 3:return console.log("running LowLevelAPI"),e.next=6,t.getPageIterator(1);case 6:return n=e.sent,e.next=9,n.current();case 9:return r=e.sent,e.next=12,r.getAnnots();case 12:if(null==(a=e.sent))return e.next=16,t.createIndirectArray();e.next=22;break;case 16:return a=e.sent,e.next=19,r.getSDFObj();case 19:return s=e.sent,e.next=22,s.put("Annots",a);case 22:return e.next=24,t.createIndirectDict();case 24:return o=e.sent,e.next=27,o.putName("Subtype","Text");case 27:return e.next=29,o.putBool("Open",!0);case 29:return e.next=31,o.putString("Contents","The quick brown fox ate the lazy mouse.");case 31:return e.next=33,o.putRect("Rect",266,116,430,204);case 33:return e.next=35,a.pushBack(o);case 35:return e.next=37,t.createIndirectDict();case 37:return c=e.sent,e.next=40,c.putName("Subtype","Link");case 40:return e.t0=PDFNet.Destination,e.next=43,t.getPage(2);case 43:return e.t1=e.sent,e.next=46,e.t0.createFit.call(e.t0,e.t1);case 46:return i=e.sent,e.t2=c,e.next=50,i.getSDFObj();case 50:return e.t3=e.sent,e.next=53,e.t2.put.call(e.t2,"Dest",e.t3);case 53:return e.next=55,c.putRect("Rect",85,705,503,661);case 55:return e.next=57,a.pushBack(c);case 57:return e.next=59,t.createIndirectDict();case 59:return u=e.sent,e.next=62,u.putName("Subtype","Link");case 62:return e.t4=PDFNet.Destination,e.next=65,t.getPage(3);case 65:return e.t5=e.sent,e.next=68,e.t4.createFit.call(e.t4,e.t5);case 68:return l=e.sent,e.t6=u,e.next=72,l.getSDFObj();case 72:return e.t7=e.sent,e.next=75,e.t6.put.call(e.t6,"Dest",e.t7);case 75:return e.next=77,u.putRect("Rect",85,638,503,594);case 77:return e.next=79,a.pushBack(u);case 79:return e.next=81,t.getPage(10);case 81:return P=e.sent,e.next=84,PDFNet.Destination.createXYZ(P,100,722,10);case 84:return x=e.sent,e.t8=u,e.next=88,x.getSDFObj();case 88:return e.t9=e.sent,e.next=91,e.t8.put.call(e.t8,"Dest",e.t9);case 91:return e.next=93,t.createIndirectDict();case 93:return D=e.sent,e.next=96,D.putName("Subtype","Link");case 96:return e.next=98,D.putRect("Rect",85,570,503,524);case 98:return e.next=100,D.putDict("A");case 100:return p=e.sent,e.next=103,p.putName("S","URI");case 103:return e.next=105,p.putString("URI","http://www.pdftron.com");case 105:return e.next=107,a.pushBack(D);case 107:return console.log("AnnotationLowLevel Done."),e.next=110,PDFNet.endDeallocateStack();case 110:e.next=115;break;case 112:e.prev=112,e.t10=e.catch(0),console.log(e.t10);case 115:case"end":return e.stop()}},e[0],this,[[0,112]])}function i(t){var n,r,a,s,o,c,i,u,l,P,x,D,p,F,N,h,d,f,g,y,w,A,S,C,k,m,v,b,L,B,R,E;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,PDFNet.startDeallocateStack();case 2:return e.next=4,t.getPage(1);case 4:return n=e.sent,console.log("Traversing all annotations in the document..."),e.next=8,t.getPage(1);case 8:return n=e.sent,r=0,e.next=12,t.getPageIterator(1);case 12:a=e.sent;case 14:return e.next=16,a.hasNext();case 16:if(e.sent)return r+=1,console.log("Page "+r+": "),e.next=21,a.current();e.next=122;break;case 21:return s=e.sent,e.next=24,s.getNumAnnots();case 24:o=e.sent,c=0;case 26:if(c<o)return e.next=29,s.getAnnot(c);e.next=118;break;case 29:return i=e.sent,e.next=32,i.isValid();case 32:if(e.sent){e.next=34;break}return e.abrupt("continue",115);case 34:return e.next=36,i.getSDFObj();case 36:return u=e.sent,e.next=39,u.get("Subtype");case 39:return l=e.sent,e.next=42,l.value();case 42:return P=e.sent,e.next=45,P.getName();case 45:return e.t0=e.sent,x="Annot Type: "+e.t0,e.next=49,i.getRect();case 49:return D=e.sent,x+=";  Position: "+D.x1+", "+D.y1+", "+D.x2+", "+D.y2,console.log(x),e.next=54,i.getType();case 54:p=e.sent,e.t1=p,e.next=e.t1===PDFNet.Annot.Type.e_Link?58:e.t1===PDFNet.Annot.Type.e_Widget?110:e.t1===PDFNet.Annot.Type.e_FileAttachment?111:112;break;case 58:return e.next=60,PDFNet.LinkAnnot.createFromAnnot(i);case 60:return F=e.sent,e.next=63,F.getAction();case 63:return N=e.sent,e.next=66,N.isValid();case 66:if(e.sent){e.next=68;break}return e.abrupt("continue",115);case 68:return e.next=70,N.getType();case 70:if(e.t2=e.sent,e.t3=PDFNet.Action.Type.e_GoTo,e.t2===e.t3)return e.next=75,N.getDest();e.next=90;break;case 75:return h=e.sent,e.next=78,h.isValid();case 78:if(e.sent){e.next=82;break}console.log("  Destination is not valid"),e.next=88;break;case 82:return e.next=84,h.getPage();case 84:return e.next=86,e.sent.getIndex();case 86:d=e.sent,console.log("  Links to: page number "+d+" in this document");case 88:e.next=109;break;case 90:return e.next=92,N.getType();case 92:if(e.t4=e.sent,e.t5=PDFNet.Action.Type.e_URI,e.t4===e.t5)return e.next=97,N.getSDFObj();e.next=109;break;case 97:return f=e.sent,e.next=100,f.get("URI");case 100:return g=e.sent,e.next=103,g.value();case 103:return y=e.sent,e.next=106,y.getAsPDFText();case 106:w=e.sent,console.log(" Links to: "+w),g.destroy();case 109:case 110:case 111:case 112:return e.abrupt("break",113);case 113:return e.next=115,l.destroy();case 115:++c,e.next=26;break;case 118:return e.next=120,a.next();case 120:e.next=14;break;case 122:return e.next=124,t.getPage(1);case 124:return n=e.sent,e.next=127,PDFNet.Action.createURI(t,"http://www.pdftron.com");case 127:return A=e.sent,S=new PDFNet.Rect(85,570,503,524),e.next=131,PDFNet.LinkAnnot.create(t,S);case 131:return C=e.sent,e.next=134,C.setAction(A);case 134:return e.next=136,n.annotPushBack(C);case 136:return e.next=138,t.getPage(3);case 138:return k=e.sent,e.t6=PDFNet.Action,e.next=142,PDFNet.Destination.createFitH(k,0);case 142:return e.t7=e.sent,e.next=145,e.t6.createGoto.call(e.t6,e.t7);case 145:return m=e.sent,e.next=148,PDFNet.LinkAnnot.create(t,new PDFNet.Rect(85,458,503,502));case 148:return F=e.sent,e.next=151,F.setAction(m);case 151:return e.next=153,PDFNet.AnnotBorderStyle.create(PDFNet.AnnotBorderStyle.Style.e_solid,3,0,0);case 153:return v=e.sent,F.setBorderStyle(v,!1),e.next=157,PDFNet.ColorPt.init(0,0,1,0);case 157:return b=e.sent,e.next=160,F.setColorDefault(b);case 160:return e.next=162,n.annotPushBack(F);case 162:return e.next=164,PDFNet.RubberStampAnnot.create(t,new PDFNet.Rect(30,30,300,200));case 164:return L=e.sent,e.next=167,L.setIconName("Draft");case 167:return e.next=169,n.annotPushBack(L);case 169:return e.next=171,PDFNet.InkAnnot.create(t,new PDFNet.Rect(110,10,300,200));case 171:return B=e.sent,R=new PDFNet.Point(110,10),e.next=175,B.setPoint(0,0,R);case 175:return R.x=150,R.y=50,e.next=179,B.setPoint(0,1,R);case 179:return R.x=190,R.y=60,e.next=183,B.setPoint(0,2,R);case 183:return R.x=180,R.y=90,e.next=187,B.setPoint(1,0,R);case 187:return R.x=190,R.y=95,e.next=191,B.setPoint(1,1,R);case 191:return R.x=200,R.y=100,e.next=195,B.setPoint(1,2,R);case 195:return R.x=166,R.y=86,e.next=199,B.setPoint(2,0,R);case 199:return R.x=196,R.y=96,e.next=203,B.setPoint(2,1,R);case 203:return R.x=221,R.y=121,e.next=207,B.setPoint(2,2,R);case 207:return R.x=288,R.y=188,e.next=211,B.setPoint(2,3,R);case 211:return e.next=213,PDFNet.ColorPt.init(0,1,1,0);case 213:return E=e.sent,e.next=216,B.setColor(E,3);case 216:return n.annotPushBack(B),e.next=219,PDFNet.endDeallocateStack();case 219:case"end":return e.stop()}},e[1],this)}function u(t){var n,r,a,s,o,c,i,u,l,P,x,D,p,F,N,h,d,f,g,y,w,A,S,C,k,m,v,b,L,B,R,E,_,I,T,O,j,G,V;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,PDFNet.startDeallocateStack();case 2:return e.next=4,PDFNet.ElementWriter.create();case 4:return n=e.sent,e.next=7,PDFNet.ElementBuilder.create();case 7:return r=e.sent,e.next=10,t.pageCreate(new PDFNet.Rect(0,0,600,600));case 10:return s=e.sent,t.pagePushBack(s),n.beginOnPage(s,PDFNet.ElementWriter.WriteMode.e_overlay,!1),n.end(),e.next=16,PDFNet.FreeTextAnnot.create(t,new PDFNet.Rect(10,400,160,570));case 16:return o=e.sent,e.next=19,o.setContents("\n\nSome swift brown fox snatched a gray hare out of the air by freezing it with an angry glare.\n\nAha!\n\nAnd there was much rejoicing!");case 19:return e.next=21,PDFNet.AnnotBorderStyle.create(PDFNet.AnnotBorderStyle.Style.e_solid,1,10,20);case 21:return c=e.sent,e.next=24,o.setBorderStyle(c,!0);case 24:return e.next=26,o.setQuaddingFormat(0);case 26:return e.next=28,s.annotPushBack(o);case 28:return e.next=30,o.refreshAppearance();case 30:return e.next=32,PDFNet.FreeTextAnnot.create(t,new PDFNet.Rect(100,100,350,500));case 32:return o=e.sent,e.next=35,o.setContentRect(new PDFNet.Rect(200,200,350,500));case 35:return e.next=37,o.setContents("\n\nSome swift brown fox snatched a gray hare out of the air by freezing it with an angry glare.\n\nAha!\n\nAnd there was much rejoicing!");case 37:return e.next=39,o.setCalloutLinePoints(new PDFNet.Point(200,300),new PDFNet.Point(150,290),new PDFNet.Point(110,110));case 39:return e.next=41,PDFNet.AnnotBorderStyle.create(PDFNet.AnnotBorderStyle.Style.e_solid,1,10,20);case 41:return c=e.sent,e.next=44,o.setBorderStyle(c,!0);case 44:return e.next=46,o.setEndingStyle(PDFNet.LineAnnot.EndingStyle.e_ClosedArrow);case 46:return e.next=48,PDFNet.ColorPt.init(0,1,0,0);case 48:return i=e.sent,e.next=51,o.setColorDefault(i);case 51:return e.next=53,o.setQuaddingFormat(1);case 53:return e.next=55,s.annotPushBack(o);case 55:return e.next=57,o.refreshAppearance();case 57:return e.next=59,PDFNet.FreeTextAnnot.create(t,new PDFNet.Rect(400,10,550,400));case 59:return o=e.sent,e.next=62,o.setContents("\n\nSome swift brown fox snatched a gray hare out of the air by freezing it with an angry glare.\n\nAha!\n\nAnd there was much rejoicing!");case 62:return e.next=64,PDFNet.AnnotBorderStyle.create(PDFNet.AnnotBorderStyle.Style.e_solid,1,10,20);case 64:return c=e.sent,e.next=67,o.setBorderStyle(c,!0);case 67:return e.next=69,PDFNet.ColorPt.init(0,0,1,0);case 69:return u=e.sent,e.next=72,o.setColorDefault(u);case 72:return e.next=74,o.setOpacity(.2);case 74:return e.next=76,o.setQuaddingFormat(2);case 76:return e.next=78,s.annotPushBack(o);case 78:return e.next=80,o.refreshAppearance();case 80:return e.next=82,t.pageCreate(new PDFNet.Rect(0,0,600,600));case 82:return l=e.sent,t.pagePushBack(l),e.next=86,n.beginOnPage(l,PDFNet.ElementWriter.WriteMode.e_overlay,!1);case 86:return e.next=88,r.reset(new PDFNet.GState("0"));case 88:return e.next=90,n.end();case 90:return e.next=92,PDFNet.LineAnnot.create(t,new PDFNet.Rect(250,250,400,400));case 92:return P=e.sent,e.next=95,P.setStartPoint(new PDFNet.Point(350,270));case 95:return e.next=97,P.setEndPoint(new PDFNet.Point(260,370));case 97:return e.next=99,P.setStartStyle(PDFNet.LineAnnot.EndingStyle.e_Square);case 99:return e.next=101,P.setEndStyle(PDFNet.LineAnnot.EndingStyle.e_Circle);case 101:return e.next=103,PDFNet.ColorPt.init(.3,.5,0,0);case 103:return x=e.sent,e.next=106,P.setColor(x,3);case 106:return e.next=108,P.setContents("Dashed Captioned");case 108:return e.next=110,P.setShowCaption(!0);case 110:return e.next=112,P.setCapPos(PDFNet.LineAnnot.CapPos.e_Top);case 112:return D=new Float64Array([2,2]),e.next=115,PDFNet.AnnotBorderStyle.createWithDashPattern(PDFNet.AnnotBorderStyle.Style.e_dashed,2,0,0,D);case 115:return p=e.sent,P.setBorderStyle(p,!1),P.refreshAppearance(),l.annotPushBack(P),e.next=121,PDFNet.LineAnnot.create(t,new PDFNet.Rect(347,377,600,600));case 121:return P=e.sent,e.next=124,P.setStartPoint(new PDFNet.Point(385,410));case 124:return e.next=126,P.setEndPoint(new PDFNet.Point(540,555));case 126:return e.next=128,P.setStartStyle(PDFNet.LineAnnot.EndingStyle.e_Circle);case 128:return e.next=130,P.setEndStyle(PDFNet.LineAnnot.EndingStyle.e_OpenArrow);case 130:return e.next=132,PDFNet.ColorPt.init(1,0,0,0);case 132:return u=e.sent,e.next=135,P.setColor(u,3);case 135:return e.next=137,PDFNet.ColorPt.init(0,1,0,0);case 137:return i=e.sent,e.next=140,P.setInteriorColor(i,3);case 140:return e.next=142,P.setContents("Inline Caption");case 142:return e.next=144,P.setShowCaption(!0);case 144:return e.next=146,P.setCapPos(PDFNet.LineAnnot.CapPos.e_Inline);case 146:return e.next=148,P.setLeaderLineExtensionLength(-4);case 148:return e.next=150,P.setLeaderLineLength(-12);case 150:return e.next=152,P.setLeaderLineOffset(2);case 152:return e.next=154,P.refreshAppearance();case 154:return l.annotPushBack(P),e.next=157,PDFNet.LineAnnot.create(t,new PDFNet.Rect(10,400,200,600));case 157:return P=e.sent,e.next=160,P.setStartPoint(new PDFNet.Point(25,426));case 160:return e.next=162,P.setEndPoint(new PDFNet.Point(180,555));case 162:return e.next=164,P.setStartStyle(PDFNet.LineAnnot.EndingStyle.e_Circle);case 164:return e.next=166,P.setEndStyle(PDFNet.LineAnnot.EndingStyle.e_Square);case 166:return e.next=168,PDFNet.ColorPt.init(0,0,1,0);case 168:return F=e.sent,e.next=171,P.setColor(F,3);case 171:return e.next=173,PDFNet.ColorPt.init(1,0,0,0);case 173:return u=e.sent,e.next=176,P.setInteriorColor(u,3);case 176:return e.next=178,P.setContents("Offset Caption");case 178:return e.next=180,P.setShowCaption(!0);case 180:return e.next=182,P.setCapPos(PDFNet.LineAnnot.CapPos.e_Top);case 182:return e.next=184,P.setTextHOffset(-60);case 184:return e.next=186,P.setTextVOffset(10);case 186:return e.next=188,P.refreshAppearance();case 188:return l.annotPushBack(P),e.next=191,PDFNet.LineAnnot.create(t,new PDFNet.Rect(200,10,400,70));case 191:return(P=e.sent).setStartPoint(new PDFNet.Point(220,25)),P.setEndPoint(new PDFNet.Point(370,60)),P.setStartStyle(PDFNet.LineAnnot.EndingStyle.e_Butt),P.setEndStyle(PDFNet.LineAnnot.EndingStyle.e_OpenArrow),e.t0=P,e.next=199,PDFNet.ColorPt.init(0,0,1);case 199:return e.t1=e.sent,e.t0.setColor.call(e.t0,e.t1,3),P.setContents("Regular Caption"),P.setShowCaption(!0),P.setCapPos(PDFNet.LineAnnot.CapPos.e_Top),e.next=206,P.refreshAppearance();case 206:return l.annotPushBack(P),e.next=209,PDFNet.LineAnnot.create(t,new PDFNet.Rect(200,70,400,130));case 209:return(P=e.sent).setStartPoint(new PDFNet.Point(220,111)),P.setEndPoint(new PDFNet.Point(370,78)),P.setStartStyle(PDFNet.LineAnnot.EndingStyle.e_Circle),P.setEndStyle(PDFNet.LineAnnot.EndingStyle.e_Diamond),P.setContents("Circle to Diamond"),e.t2=P,e.next=218,PDFNet.ColorPt.init(0,0,1);case 218:return e.t3=e.sent,e.t2.setColor.call(e.t2,e.t3,3),e.t4=P,e.next=223,PDFNet.ColorPt.init(0,1,0);case 223:return e.t5=e.sent,e.t4.setInteriorColor.call(e.t4,e.t5,3),P.setShowCaption(!0),P.setCapPos(PDFNet.LineAnnot.CapPos.e_Top),P.refreshAppearance(),l.annotPushBack(P),e.next=231,PDFNet.LineAnnot.create(t,new PDFNet.Rect(10,100,160,200));case 231:return(P=e.sent).setStartPoint(new PDFNet.Point(15,110)),P.setEndPoint(new PDFNet.Point(150,190)),P.setStartStyle(PDFNet.LineAnnot.EndingStyle.e_Slash),P.setEndStyle(PDFNet.LineAnnot.EndingStyle.e_ClosedArrow),P.setContents("Slash to CArrow"),e.t6=P,e.next=240,PDFNet.ColorPt.init(1,0,0);case 240:return e.t7=e.sent,e.t6.setColor.call(e.t6,e.t7,3),e.t8=P,e.next=245,PDFNet.ColorPt.init(0,1,1);case 245:return e.t9=e.sent,e.t8.setInteriorColor.call(e.t8,e.t9,3),P.setShowCaption(!0),P.setCapPos(PDFNet.LineAnnot.CapPos.e_Top),P.refreshAppearance(),l.annotPushBack(P),e.next=253,PDFNet.LineAnnot.create(t,new PDFNet.Rect(270,270,570,433));case 253:return(P=e.sent).setStartPoint(new PDFNet.Point(300,400)),P.setEndPoint(new PDFNet.Point(550,300)),P.setStartStyle(PDFNet.LineAnnot.EndingStyle.e_RClosedArrow),P.setEndStyle(PDFNet.LineAnnot.EndingStyle.e_ROpenArrow),P.setContents("ROpen & RClosed arrows"),e.t10=P,e.next=262,PDFNet.ColorPt.init(0,0,1);case 262:return e.t11=e.sent,e.t10.setColor.call(e.t10,e.t11,3),e.t12=P,e.next=267,PDFNet.ColorPt.init(0,1,0);case 267:return e.t13=e.sent,e.t12.setInteriorColor.call(e.t12,e.t13,3),P.setShowCaption(!0),P.setCapPos(PDFNet.LineAnnot.CapPos.e_Top),P.refreshAppearance(),l.annotPushBack(P),e.next=275,PDFNet.LineAnnot.create(t,new PDFNet.Rect(195,395,205,505));case 275:return(P=e.sent).setStartPoint(new PDFNet.Point(200,400)),P.setEndPoint(new PDFNet.Point(200,500)),P.refreshAppearance(),l.annotPushBack(P),e.next=282,PDFNet.LineAnnot.create(t,new PDFNet.Rect(55,299,150,301));case 282:return(P=e.sent).setStartPoint(new PDFNet.Point(55,300)),P.setEndPoint(new PDFNet.Point(155,300)),P.setStartStyle(PDFNet.LineAnnot.EndingStyle.e_Circle),P.setEndStyle(PDFNet.LineAnnot.EndingStyle.e_Circle),P.setContents("Caption that's longer than its line."),e.t14=P,e.next=291,PDFNet.ColorPt.init(1,0,1);case 291:return e.t15=e.sent,e.t14.setColor.call(e.t14,e.t15,3),e.t16=P,e.next=296,PDFNet.ColorPt.init(0,1,0);case 296:return e.t17=e.sent,e.t16.setInteriorColor.call(e.t16,e.t17,3),P.setShowCaption(!0),P.setCapPos(PDFNet.LineAnnot.CapPos.e_Top),P.refreshAppearance(),l.annotPushBack(P),e.next=304,PDFNet.LineAnnot.create(t,new PDFNet.Rect(300,200,390,234));case 304:return(P=e.sent).setStartPoint(new PDFNet.Point(310,210)),P.setEndPoint(new PDFNet.Point(380,220)),e.t18=P,e.next=310,PDFNet.ColorPt.init(0,0,0);case 310:return e.t19=e.sent,e.t18.setColor.call(e.t18,e.t19,3),P.refreshAppearance(),l.annotPushBack(P),e.next=316,t.pageCreate(new PDFNet.Rect(0,0,600,600));case 316:return N=e.sent,n.beginOnPage(N),n.end(),t.pagePushBack(N),e.next=322,PDFNet.CircleAnnot.create(t,new PDFNet.Rect(300,300,390,350));case 322:return h=e.sent,e.t20=h,e.next=326,PDFNet.ColorPt.init(0,0,0);case 326:return e.t21=e.sent,e.t20.setColor.call(e.t20,e.t21,3),h.refreshAppearance(),N.annotPushBack(h),e.next=332,PDFNet.CircleAnnot.create(t,new PDFNet.Rect(100,100,200,200));case 332:return h=e.sent,e.t22=h,e.next=336,PDFNet.ColorPt.init(0,1,0);case 336:return e.t23=e.sent,e.t22.setColor.call(e.t22,e.t23,3),e.t24=h,e.next=341,PDFNet.ColorPt.init(0,0,1);case 341:return e.t25=e.sent,e.t24.setInteriorColor.call(e.t24,e.t25,3),D=[2,4],e.t26=h,e.next=347,PDFNet.AnnotBorderStyle.createWithDashPattern(PDFNet.AnnotBorderStyle.Style.e_dashed,3,0,0,D);case 347:return e.t27=e.sent,e.t26.setBorderStyle.call(e.t26,e.t27),h.setPadding(new PDFNet.Rect(2,2,2,2)),h.refreshAppearance(),N.annotPushBack(h),e.next=354,PDFNet.SquareAnnot.create(t,new PDFNet.Rect(10,200,80,300));case 354:return d=e.sent,e.t28=d,e.next=358,PDFNet.ColorPt.init(0,0,0);case 358:return e.t29=e.sent,e.t28.setColor.call(e.t28,e.t29,3),d.refreshAppearance(),N.annotPushBack(d),e.next=364,PDFNet.SquareAnnot.create(t,new PDFNet.Rect(500,200,580,300));case 364:return d=e.sent,e.t30=d,e.next=368,PDFNet.ColorPt.init(1,0,0);case 368:return e.t31=e.sent,e.t30.setColor.call(e.t30,e.t31,3),e.t32=d,e.next=373,PDFNet.ColorPt.init(0,1,1);case 373:return e.t33=e.sent,e.t32.setInteriorColor.call(e.t32,e.t33,3),D=[4,2],e.t34=d,e.next=379,PDFNet.AnnotBorderStyle.createWithDashPattern(PDFNet.AnnotBorderStyle.Style.e_dashed,6,0,0,D);case 379:return e.t35=e.sent,e.t34.setBorderStyle.call(e.t34,e.t35),d.setPadding(new PDFNet.Rect(4,4,4,4)),d.refreshAppearance(),N.annotPushBack(d),e.next=386,PDFNet.PolygonAnnot.create(t,new PDFNet.Rect(5,500,125,590));case 386:return f=e.sent,e.t36=f,e.next=390,PDFNet.ColorPt.init(1,0,0);case 390:return e.t37=e.sent,e.t36.setColor.call(e.t36,e.t37,3),e.t38=f,e.next=395,PDFNet.ColorPt.init(1,1,0);case 395:return e.t39=e.sent,e.t38.setInteriorColor.call(e.t38,e.t39,3),f.setVertex(0,new PDFNet.Point(12,510)),f.setVertex(1,new PDFNet.Point(100,510)),f.setVertex(2,new PDFNet.Point(100,555)),f.setVertex(3,new PDFNet.Point(35,544)),e.next=403,PDFNet.AnnotBorderStyle.create(PDFNet.AnnotBorderStyle.Style.e_solid,4,0,0);case 403:return g=e.sent,f.setBorderStyle(g),f.setPadding(new PDFNet.Rect(4,4,4,4)),f.refreshAppearance(),N.annotPushBack(f),e.next=410,PDFNet.PolyLineAnnot.create(t,new PDFNet.Rect(400,10,500,90));case 410:return f=e.sent,e.t40=f,e.next=414,PDFNet.ColorPt.init(1,0,0);case 414:return e.t41=e.sent,e.t40.setColor.call(e.t40,e.t41,3),e.t42=f,e.next=419,PDFNet.ColorPt.init(0,1,0);case 419:return e.t43=e.sent,e.t42.setInteriorColor.call(e.t42,e.t43,3),f.setVertex(0,new PDFNet.Point(405,20)),f.setVertex(1,new PDFNet.Point(440,40)),f.setVertex(2,new PDFNet.Point(410,60)),f.setVertex(3,new PDFNet.Point(470,80)),e.t44=f,e.next=428,PDFNet.AnnotBorderStyle.create(PDFNet.AnnotBorderStyle.Style.e_solid,2,0,0);case 428:return e.t45=e.sent,e.t44.setBorderStyle.call(e.t44,e.t45),f.setPadding(new PDFNet.Rect(4,4,4,4)),f.setStartStyle(PDFNet.LineAnnot.EndingStyle.e_RClosedArrow),f.setEndStyle(PDFNet.LineAnnot.EndingStyle.e_ClosedArrow),f.refreshAppearance(),N.annotPushBack(f),e.next=437,PDFNet.LinkAnnot.create(t,new PDFNet.Rect(5,5,55,24));case 437:return(y=e.sent).refreshAppearance(),N.annotPushBack(y),e.next=442,t.pageCreate(new PDFNet.Rect(0,0,600,600));case 442:return w=e.sent,n.beginOnPage(w),n.end(),t.pagePushBack(w),n.beginOnPage(w),e.next=449,PDFNet.Font.create(t,PDFNet.Font.StandardType1Font.e_helvetica);case 449:return A=e.sent,e.next=452,r.createTextBeginWithFont(A,16);case 452:return(a=e.sent).setPathFill(!0),n.writeElement(a),e.next=457,r.createTextRun("Some random text on the page",A,16);case 457:return(a=e.sent).setTextMatrixEntries(1,0,0,1,100,500),n.writeElement(a),e.t46=n,e.next=463,r.createTextEnd();case 463:return e.t47=e.sent,e.t46.writeElement.call(e.t46,e.t47),n.end(),e.next=468,PDFNet.HighlightAnnot.create(t,new PDFNet.Rect(100,490,150,515));case 468:return S=e.sent,e.t48=S,e.next=472,PDFNet.ColorPt.init(0,1,0);case 472:return e.t49=e.sent,e.t48.setColor.call(e.t48,e.t49,3),S.refreshAppearance(),w.annotPushBack(S),e.next=478,PDFNet.SquigglyAnnot.create(t,new PDFNet.Rect(100,450,250,600));case 478:return(d=e.sent).setQuadPoint(0,PDFNet.QuadPoint(122,455,240,545,230,595,101,500)),d.refreshAppearance(),w.annotPushBack(d),e.next=484,PDFNet.CaretAnnot.create(t,new PDFNet.Rect(100,40,129,69));case 484:return C=e.sent,e.t50=C,e.next=488,PDFNet.ColorPt.init(0,0,1);case 488:return e.t51=e.sent,e.t50.setColor.call(e.t50,e.t51,3),C.setSymbol("P"),C.refreshAppearance(),w.annotPushBack(C),e.next=495,t.pageCreate(new PDFNet.Rect(0,0,600,600));case 495:return k=e.sent,n.beginOnPage(k),n.end(),t.pagePushBack(k),e.next=501,PDFNet.FileSpec.create(t,"../TestFiles/butterfly.png",!1);case 501:return m=e.sent,e.next=504,t.pageCreate(new PDFNet.Rect(0,0,600,600));case 504:v=e.sent,n.beginOnPage(v),n.end(),t.pagePushBack(v),b=0;case 509:if(!(b<2)){e.next=549;break}L=0;case 511:if(!(L<100)){e.next=546;break}if(L>PDFNet.FileAttachmentAnnot.Icon.e_Tag){e.next=524;break}return e.next=515,PDFNet.FileAttachmentAnnot.createWithFileSpec(t,new PDFNet.Rect(50+50*L,100,70+50*L,120),m,L);case 515:if(B=e.sent,b)return e.t52=B,e.next=520,PDFNet.ColorPt.init(1,1,0);e.next=522;break;case 520:e.t53=e.sent,e.t52.setColor.call(e.t52,e.t53);case 522:B.refreshAppearance(),0===b?k.annotPushBack(B):v.annotPushBack(B);case 524:if(L>PDFNet.TextAnnot.Icon.e_Note)return e.abrupt("break",546);e.next=526;break;case 526:return e.next=528,PDFNet.TextAnnot.create(t,new PDFNet.Rect(10+50*L,200,30+50*L,220));case 528:return(R=e.sent).setIcon(L),e.t54=R,e.next=533,R.getIconName();case 533:if(e.t55=e.sent,e.t54.setContents.call(e.t54,e.t55),b)return e.t56=R,e.next=539,PDFNet.ColorPt.init(1,1,0);e.next=541;break;case 539:e.t57=e.sent,e.t56.setColor.call(e.t56,e.t57);case 541:R.refreshAppearance(),0===b?k.annotPushBack(R):v.annotPushBack(R);case 543:L++,e.next=511;break;case 546:++b,e.next=509;break;case 549:return e.next=551,PDFNet.TextAnnot.create(t,new PDFNet.Rect(10,20,30,40));case 551:return(R=e.sent).setIconName("UserIcon"),R.setContents("User defined icon, unrecognized by appearance generator"),e.t58=R,e.next=557,PDFNet.ColorPt.init(0,1,0);case 557:return e.t59=e.sent,e.t58.setColor.call(e.t58,e.t59),R.refreshAppearance(),v.annotPushBack(R),e.next=563,PDFNet.InkAnnot.create(t,new PDFNet.Rect(100,400,200,550));case 563:return E=e.sent,e.t60=E,e.next=567,PDFNet.ColorPt.init(0,0,1);case 567:return e.t61=e.sent,e.t60.setColor.call(e.t60,e.t61),E.setPoint(1,3,new PDFNet.Point(220,505)),E.setPoint(1,0,new PDFNet.Point(100,490)),E.setPoint(0,1,new PDFNet.Point(120,410)),E.setPoint(0,0,new PDFNet.Point(100,400)),E.setPoint(1,2,new PDFNet.Point(180,490)),E.setPoint(1,1,new PDFNet.Point(140,440)),e.t62=E,e.next=578,PDFNet.AnnotBorderStyle.create(PDFNet.AnnotBorderStyle.Style.e_solid,3,0,0);case 578:return e.t63=e.sent,e.t62.setBorderStyle.call(e.t62,e.t63),E.refreshAppearance(),v.annotPushBack(E),e.next=584,t.pageCreate(new PDFNet.Rect(0,0,600,600));case 584:return _=e.sent,n.beginOnPage(_),n.end(),t.pagePushBack(_),e.next=590,PDFNet.SoundAnnot.create(t,new PDFNet.Rect(100,500,120,520));case 590:return I=e.sent,e.t64=I,e.next=594,PDFNet.ColorPt.init(1,1,0);case 594:return e.t65=e.sent,e.t64.setColor.call(e.t64,e.t65),I.setIcon(PDFNet.SoundAnnot.Icon.e_Speaker),I.refreshAppearance(),_.annotPushBack(I),e.next=601,PDFNet.SoundAnnot.create(t,new PDFNet.Rect(200,500,220,520));case 601:return I=e.sent,e.t66=I,e.next=605,PDFNet.ColorPt.init(1,1,0);case 605:return e.t67=e.sent,e.t66.setColor.call(e.t66,e.t67),I.setIcon(PDFNet.SoundAnnot.Icon.e_Mic),I.refreshAppearance(),_.annotPushBack(I),e.next=612,t.pageCreate(new PDFNet.Rect(0,0,600,600));case 612:T=e.sent,n.beginOnPage(T),n.end(),t.pagePushBack(T),b=0;case 617:if(!(b<2)){e.next=641;break}O=5,j=520,G=PDFNet.RubberStampAnnot.Icon.e_Approved;case 621:if(G<=PDFNet.RubberStampAnnot.Icon.e_Draft)return e.next=624,PDFNet.RubberStampAnnot.create(t,new PDFNet.Rect(1,1,100,100));e.next=638;break;case 624:return(V=e.sent).setIcon(G),e.t68=V,e.next=629,V.getIconName();case 629:e.t69=e.sent,e.t68.setContents.call(e.t68,e.t69),V.setRect(new PDFNet.Rect(O,j,O+100,j+25)),(j-=100)<0&&(j=520,O+=200),0===b||(T.annotPushBack(V),V.refreshAppearance());case 635:G++,e.next=621;break;case 638:++b,e.next=617;break;case 641:return e.next=643,PDFNet.RubberStampAnnot.create(t,new PDFNet.Rect(400,5,550,45));case 643:return(V=e.sent).setIconName("UserStamp"),V.setContents("User defined stamp"),T.annotPushBack(V),V.refreshAppearance(),e.next=650,PDFNet.endDeallocateStack();case 650:case"end":return e.stop()}},e[2],this)}function t(){var t,n,r,a,s,o;return regeneratorRuntime.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,console.log("Beginning Annotation Test. This test will add different annotations to PDF documents."),t=0,"../TestFiles/",e.next=6,PDFNet.PDFDoc.createFromURL("../TestFiles/numbered.pdf");case 6:return(n=e.sent).initSecurityHandler(),n.lock(),console.log("PDFNet and PDF document initialized and locked"),e.delegateYield(c(n),"t0",11);case 11:return e.next=13,n.saveMemoryBuffer(PDFNet.SDFDoc.SaveOptions.e_linearized);case 13:return r=e.sent,saveBufferAsPDFDoc(r,"annotation_testLowLevel.pdf"),e.next=17,n.getPage(1);case 17:return e.sent,e.delegateYield(i(n),"t1",19);case 19:return e.next=21,n.saveMemoryBuffer(PDFNet.SDFDoc.SaveOptions.e_linearized);case 21:return a=e.sent,saveBufferAsPDFDoc(a,"annotation_testHighLevel.pdf"),e.next=25,PDFNet.PDFDoc.create();case 25:return(s=e.sent).lock(),e.delegateYield(u(s),"t2",28);case 28:return e.next=30,s.saveMemoryBuffer(PDFNet.SDFDoc.SaveOptions.e_linearized);case 30:return o=e.sent,saveBufferAsPDFDoc(o,"new_annot_test_api.pdf"),console.log("Done."),e.abrupt("return",t);case 36:e.prev=36,e.t3=e.catch(0),console.log(e.t3);case 39:case"end":return e.stop()}},e[3],this,[[0,36]])}PDFNet.runGeneratorWithCleanup(t(),window.sampleL)}}();