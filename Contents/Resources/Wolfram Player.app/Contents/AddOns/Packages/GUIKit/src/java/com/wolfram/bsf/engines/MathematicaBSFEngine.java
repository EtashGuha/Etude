/*
 * @(#)MathematicaBSFEngine.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.engines;

import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Stack;
import java.util.Vector;
import java.net.URL;

// BSF import switch
import org.apache.bsf.BSFException;
import org.apache.bsf.BSFManager;
import org.apache.bsf.BSFDeclaredBean;
import org.apache.bsf.util.BSFEngineImpl;
//

import com.wolfram.jlink.*;
import com.wolfram.bsf.util.*;
import com.wolfram.bsf.util.event.MathematicaEventProcessor;


/**
 * MathematicaBSFEngine is the interface to Mathematica from the
 * Bean Scripting Framework.
 * <p>
 */
public class MathematicaBSFEngine extends BSFEngineImpl {

  /* NOTE! that even though we have subclasses for BSFManager
    specifically for running the engine within Mathematica, this engine
    also needs to work external to Mathematica with the default BSFManager
     with appropriate defaults within this class to fallback on
  */

  private String contextName = null;
  // This will be set to the previous $Context before entering the Script context
  private String externalContextName = null;
  
  // We record $ContextPath on intial setting of link but we may
  // never use this yet
  // private String contextPathString = null;

  protected HashSet declaredSymbols = new HashSet();

  private static int contextCount = 0;
  
  public static final String ID_PRIVATE_PREFIX = "bsf_engine_";
  
  /* These are useful for any bsf use of the engine such as within ant */
  public static final String ID_USESINGLEKERNELLINK = ID_PRIVATE_PREFIX + "Mathematica_UseSingleKernelLink";
  public static final String ID_KERNELLINK = ID_PRIVATE_PREFIX + "Mathematica_KernelLink";
  public static final String ID_KERNELLINKCOMMANDLINE = ID_PRIVATE_PREFIX + "Mathematica_KernelLinkCommandLine";
  public static final String ID_KERNELLINKVERSION = ID_PRIVATE_PREFIX + "Mathematica_KernelLinkVersion";
  public static final String ID_CONTEXT = ID_PRIVATE_PREFIX + "Mathematica_Context";
  
  public static final String ID_DEFAULTFRAMEIMAGE = ID_PRIVATE_PREFIX + "DefaultFrameImage";
  
  /* These are really GUI private useful though an advanced dev
     might want to know context, rootObject etc */
  public static final String ID_ENDMODALSCRIPT = ID_PRIVATE_PREFIX + "EndModalScriptString";
  public static final String ID_SCRIPTCONTEXTURL = ID_PRIVATE_PREFIX + "ScriptContextURL";
  public static final String ID_SCRIPTURLSTACK = ID_PRIVATE_PREFIX + "ScriptUrlStack";
  public static final String ID_INCLUDEDSCRIPTCONTEXTS = ID_PRIVATE_PREFIX + "IncludedScriptContexts";
  
  public static final String ID_DRIVER = ID_PRIVATE_PREFIX + "Driver";
  public static final String DRIVER_SYMBOL = "Private`BSF`driver";
	public static final String ENGINE_SYMBOL = "Private`BSF`engine";
  public static final String SCRIPTCONTEXTPATH_SYMBOL = "Private`BSF`$ScriptContextPath";
  public static final String SCRIPTEXTERNALCONTEXTPATH_SYMBOL = "Private`BSF`$ScriptExternalContextPath";
  
  private boolean useEndModalSymbol = false;
  
  private static final String CONTEXT_BASE = "GUIKit`";
  
  private static final String PRIVATE_MATHEMATICABSF_CONTEXT = CONTEXT_BASE + "Private`MathematicaBSFEngine`";
	private static final String PRIVATE_EVAL_CONTEXT = PRIVATE_MATHEMATICABSF_CONTEXT + "PrivateEvaluate`";
	
  private static final String SCRIPTEVAL_FUNC = PRIVATE_MATHEMATICABSF_CONTEXT + "ScriptEvaluate";
  private static final String SCRIPTEVAL_ENDMODAL_FUNC = PRIVATE_MATHEMATICABSF_CONTEXT + "ScriptEvaluateEndModal";
  private static final String SCRIPTFILEEVAL_FUNC = PRIVATE_MATHEMATICABSF_CONTEXT + "ScriptFileEvaluate";
  private static final String SCRIPTFILEEVAL_ENDMODAL_FUNC = PRIVATE_MATHEMATICABSF_CONTEXT + "ScriptFileEvaluateEndModal";
  
	private static final String UNDECLARESYMS_FUNC = PRIVATE_MATHEMATICABSF_CONTEXT + "UndeclareSymbols";
	private static final String EXPRTOXMLSTRING_FUNC = PRIVATE_MATHEMATICABSF_CONTEXT + "RequestExprToXMLString";
  private static final String EXPRFILETOXMLSTRING_FUNC = PRIVATE_MATHEMATICABSF_CONTEXT + "RequestExprFileToXMLString";
  private static final String ENDMODAL_RESULT_STORAGE = PRIVATE_MATHEMATICABSF_CONTEXT + "EndModalResult";

  // TODO need to decide if this should be `Private and have
  // children environments create engines with nested context
  // naming? Will allow one to know parents.
  // ALSO we make need to engine lock on parent environment hashing
  // since multiple engines can exist for one definition set
  // and will children clean up properly and also not block
  // with engine locking??
  public static final String CONTEXT_SUFFIX = CONTEXT_BASE + "Private`Script";
  public static final String ID_SCRIPTEVALUATOR = "ScriptEvaluator";
    
  // Even with a private context for the script we want the bsf symbol private from
  // that to not conflict with user symbols and we will have functions wrapping access to
  // this symbol
  public static final String BSF_SYMBOL = "Private`BSF`bsf";
  public static final String BEAN_SYMBOL = "Bean";
  private static final String ENDMODAL_SOURCE = "<endmodal>";
  private static final String DEFAULT_NEWKERNEL_VERSION = "5.1";

  private MathematicaBSFManager mathMgr = null;
  
  private MathematicaBSFFunctions functions = null;

  // Default to a thirty second MathLink connection timeout
	private static int connectTimeout = 30000;

  private static int singleKernelLinkCount = 0;
  private static KernelLink singleKernelLink = null;

  private static LinkClassLoaderHandler classLoaderHandler;
  
  private Integer sharedKernelToken = null;
  // Starting with 6.0 an integer token is not returned or used
  // but we use this token to know we called share kernel for 
  // consistency with logic from pre-6.0 systems
  private static final Integer SHARED_SHAREKERNEL_TOKEN = new Integer(-1);
  
  private boolean didShareFrontEnd = false;

  private boolean engineCreatedLink = false;
  private boolean startedAsStdLink = true;
  
  private EvalPacketListener evalPacketListener;

  private Double mathematicaVersionNumber = null;
  
	private static Hashtable engineLocksHash = new Hashtable(4, 0.75F);
	
	private static synchronized Object getEngineLock() {
    KernelLink l = StdLink.getLink();
		Object lock = engineLocksHash.get(l);
		if (lock == null) {
			lock = new Object();
			engineLocksHash.put(l, lock);
			}
		return lock;
		}
  
  // This could be null if kernel has not executed
  // anything yet
  public Double getMathematicaVersionNumber() {
    return mathematicaVersionNumber;
    }
    
  public static LinkClassLoaderHandler getClassLoaderHandler() {
    if (classLoaderHandler == null) {
      if (com.wolfram.jlink.Utils.getJLinkVersion().startsWith("2.")) classLoaderHandler = new JLink2ClassLoaderHandler();
      else classLoaderHandler = new JLink3ClassLoaderHandler();
      }
    return classLoaderHandler;
    }

  public ClassLoader getClassLoader() {
  	return getClassLoaderHandler().getClassLoader();
  	}
	public Class classFromBytes(String className, byte[] bytes) {
		return getClassLoaderHandler().classFromBytes(className, bytes);
		}
  public String[] getClassPath() {
    return getClassLoaderHandler().getClassPath();
    }
	public Class classLoad(String name) throws ClassNotFoundException {
		return getClassLoaderHandler().classLoad(name);
		}

  public MathematicaBSFManager getMathematicaBSFManager() {return mathMgr;}
  
  public boolean shareKernel() throws MathLinkException {
    boolean calledShareKernel = true;
    
    if (startedAsStdLink && sharedKernelToken == null ) {

    synchronized(getEngineLock()) {
      KernelLink ml = StdLink.getLink();
      
      // We need to make sure ShareFrontEnd will work if user scripts
      // use Notebook functions or other functionality that need
      // front end call backs

      // to consider, have a setting where we override even checking to use a front end?
      boolean useShareFrontEnd = true;

			// We now allow useShareFrontEnd on OS X because the fix for
			// a working ShareFrontEnd on OS X 10 is in JLink 2.1.0
			// You MUST use JLink 2.1.0 for UnshareFrontEnd not to hang on OS X 10.2
			//if (Utils.isMacOSX()) {
			//	useShareFrontEnd = false;
			//	}
			
      // We need to check if a front end actually exists and if not,
      // turn off a possible request for ShareFrontEnd to use only ShareKernel
      if (useShareFrontEnd) {
        StdLink.requestTransaction();
        synchronized (ml) {
          ml.evaluate( "TrueQ[Head[$FrontEnd] =!= FrontEndObject]");
          while (ml.waitForAnswer() != MathLink.RETURNPKT) {
            ml.newPacket();
            }
          String val = ml.getSymbol();
					useShareFrontEnd = !("True".equals(val));

          if (mathMgr != null && mathMgr.getDebug())
            mathMgr.getDebugStream().println("is useShareFrontEnd: " + useShareFrontEnd);
          }
        }

      StdLink.requestTransaction();
      synchronized (ml) {

        // JLink source says use EnterXXXPacket here but Evaluate seems to work better
        // However UnshareKernel calls *must* use EnterXXXPackets
        // Note we also use Prompt->"" to not make it known to user we
        // are turning on ShareKernel here
        if (useShareFrontEnd) {
          // ShareFrontEnd handling options is new to JLink 2.1 but we are 
          // propbably going to require this minimum version
          ml.evaluate("ShareFrontEnd[Prompt->\"\"]");
          didShareFrontEnd = true;
          }
        else {
          ml.evaluate( "ShareKernel[Prompt->\"\"]");
          didShareFrontEnd = false;
          }
        while (ml.waitForAnswer() != MathLink.RETURNPKT) {
          ml.newPacket();
          }
        switch (ml.getNext()) {
          case MathLink.MLTKINT:
            sharedKernelToken = new Integer(ml.getInteger());
            break;
          case MathLink.MLTKSYM:
            String sym = ml.getSymbol();
            if (sym.equals("Null")) {
              sharedKernelToken = SHARED_SHAREKERNEL_TOKEN;
              }
            else calledShareKernel = false;
            break;
          default:
            ml.newPacket();
            break;
            }
        
        if (mathMgr != null && mathMgr.getDebug())
          mathMgr.getDebugStream().println("called " +
             (didShareFrontEnd ? "ShareFrontEnd" : "ShareKernel") + " id: " + 
                (sharedKernelToken != null ? "" + sharedKernelToken.intValue() : " failed!"));

        }
        
      }
      
      }

      return calledShareKernel;
    }

  public void unshareKernel() {
    if (sharedKernelToken == null) {
      return;
      }

    if (mathMgr != null && mathMgr.getDebug())
      mathMgr.getDebugStream().println("calling " +
        (didShareFrontEnd ? "UnshareFrontEnd" : "UnshareKernel") + " id: " + sharedKernelToken.intValue());

    synchronized(getEngineLock()) {
		  KernelLink ml = StdLink.getLink();
		  
    StdLink.requestTransaction();
    synchronized (ml) {
      try {
        // Note must use EnterTextPacket because of UnshareKernel use
        ml.putFunction("EnterTextPacket", 1);
        if (didShareFrontEnd) {
          ml.put("(UnshareFrontEnd[" + sharedKernelToken.intValue() + "]; If[KernelSharedQ[], $Line--];)");
          }
        else {
          ml.put("(UnshareKernel[" + sharedKernelToken.intValue() + "]; If[KernelSharedQ[], $Line--];)");
          }
        ml.discardAnswer();
        }
      catch (MathLinkException me) {
        me.printStackTrace();
        }
      }

    sharedKernelToken = null;
		}
		
    }

  public void log(String message) {
    if (mathMgr != null && mathMgr.getDebug())
      mathMgr.getDebugStream().println(message);
    }

  public void requestDeclareSymbol(String sym, Object obj) {
    if (obj == null) return;
    if (mathMgr != null && mathMgr.getDebug())
      mathMgr.getDebugStream().println("calling requestDeclareSymbol... " + sym);
    try {
      declareSymbol(contextName + sym, obj);
      }
    catch (BSFException be) {
      be.printStackTrace();
      }
    }

  public String resolveMathematicaFile(String resolveFunction, String file, String parentDirectory) {
    String result = null;
    if (file == null) return result;

    try {
      synchronized(getEngineLock()) {
				KernelLink ml = StdLink.getLink();
				
      Stack pathStack = null;
      Object scriptPathStack = mgr.lookupBean(ID_SCRIPTURLSTACK);
      if (scriptPathStack != null && scriptPathStack instanceof Stack) {
        pathStack = (Stack)scriptPathStack;
        }
          
        StdLink.requestTransaction();
      synchronized (ml) {
        ml.putFunction("EvaluatePacket", 1);
        // TODO perhaps we wrap with AbortProtect
        int count = 1;
        if (pathStack != null) count += pathStack.size();
        if (parentDirectory != null) ++count;
        
        ml.putFunction(resolveFunction, count);
        ml.put(file);

        if (pathStack != null) {
          int stackSize = pathStack.size()-1;
          for (int i = stackSize; i >= 0; --i) {
            Object url = pathStack.elementAt(i);
            String path = null;
            if (url != null && url instanceof URL)
              path = resolveToMathematicaPath((URL)url);
            ml.putFunction("DirectoryName", 1);
              ml.put(path != null ? path : "");
            }
          }
        if (parentDirectory != null)
          ml.put(parentDirectory);
          
        ml.endPacket();
        ml.flush();
        while (ml.waitForAnswer() != MathLink.RETURNPKT) {
          ml.newPacket();
          }

        if (ml.getNext() == MathLink.MLTKSTR)
          result = ml.getString();
        else {
          result = "";
          ml.newPacket();
          }

        }
			}
      }
    catch(MathLinkException me) {
      me.printStackTrace();
      }

    if (result.equals("$Failed") || result.startsWith(resolveFunction) ||
        result.equals(""))
      result = null;

    return result;
    }

  public void requestHandleException(Exception e) throws MathLinkException {
    synchronized(getEngineLock()) {
			KernelLink ml = StdLink.getLink();
      StdLink.requestTransaction();
	    synchronized (ml) {
	      try {
	        // manually put an appropriate message and then return $Failed
	        ml.beginManual();
	        ml.putFunction("CompoundExpression", 2);
					// Since this can contain a large message string do
					// we temporarily turn off $MessagePrePrint = Short here?
					ml.putFunction("Block", 2);
					ml.putFunction("List", 1);
					ml.putFunction("Set", 2);
					ml.putSymbol("$MessagePrePrint");
					ml.putSymbol("Identity");
	        ml.putFunction("Message", 2);
	          ml.putFunction("MessageName", 2);
	            ml.putSymbol("GUIRun");
	            ml.put("err");
	          ml.put( (e.getMessage() != null) ? e.getMessage() : e.getClass().getName());
					// We need to think about whether in all cases we check for this $Failed
	        ml.putSymbol("$Failed");
	        ml.flush();
	        }
	      catch (Exception ex) {
	        ex.printStackTrace();
	        }
	      }
			}
		}

  public void requestAbort() throws MathLinkException {
 		StdLink.getLink().abortEvaluation();
    }
  
  public void requestJavaShow(Object window) throws MathLinkException {
    synchronized(getEngineLock()) {
			KernelLink ml = StdLink.getLink();
      StdLink.requestTransaction();
	    synchronized (ml) {
	      ml.putFunction("EvaluatePacket", 1);
	      // TODO perhaps we wrap with AbortProtect
	      ml.putFunction("JavaShow", 1);
	      ml.putReference(window);
	      ml.endPacket();
	      ml.flush();
	      ml.discardAnswer();
	      }
			}
    }

  public void requestReleaseJavaObject(Object obj) throws MathLinkException {
   synchronized(getEngineLock()) {
      KernelLink ml = StdLink.getLink();
      StdLink.requestTransaction();
      synchronized (ml) {
        ml.putFunction("EvaluatePacket", 1);
        // TODO perhaps we wrap with AbortProtect
        ml.putFunction("ReleaseJavaObject", 1);
        ml.putReference(obj);
        ml.endPacket();
        ml.flush();
        ml.discardAnswer();
        }
		  }
    }

  
  public String requestExprFileToXMLString(String expressionFile, String format) throws MathLinkException {
    String result = null;
    
    synchronized(getEngineLock()) {
      KernelLink ml = StdLink.getLink();
      StdLink.requestTransaction();
      synchronized (ml) {
      ml.putFunction("EvaluatePacket", 1);
        ml.putFunction(EXPRFILETOXMLSTRING_FUNC, 4);
          ml.put(expressionFile);
          ml.put(format);
          ml.put(PRIVATE_EVAL_CONTEXT);
          ml.put(PRIVATE_EVAL_CONTEXT + "*");     
        ml.endPacket();
        ml.flush();
        while (ml.waitForAnswer() != MathLink.RETURNPKT) {
          ml.newPacket();
          }
        result = ml.getString();
        if (result.equals("$Failed") || result.startsWith("ExportString") || result.equals("")) 
          result = null;
        }  

      }
    return result;
    }
  
  public String requestExprToXMLString(String expressionString, String format) throws MathLinkException {
    String result = null;
    
    synchronized(getEngineLock()) {
			KernelLink ml = StdLink.getLink();
	    StdLink.requestTransaction();
			synchronized (ml) {
			ml.putFunction("EvaluatePacket", 1);
				ml.putFunction(EXPRTOXMLSTRING_FUNC, 4);
					ml.put(expressionString);
					ml.put(format);
					ml.put(PRIVATE_EVAL_CONTEXT);
					ml.put(PRIVATE_EVAL_CONTEXT + "*");			
				ml.endPacket();
				ml.flush();
				while (ml.waitForAnswer() != MathLink.RETURNPKT) {
					ml.newPacket();
					}
				result = ml.getString();
				if (result.equals("$Failed") || result.startsWith("ExportString") || result.equals("")) 
					result = null;
				}  

			}
    return result;
    }

	public Object evaluateEndModalResult() throws MathLinkException {
    Object result = null;
		Object returnScriptCode = null;

		if (mathMgr != null && mathMgr.getDebug())
			mathMgr.getDebugStream().println("calling evaluateEndModalResult...");

		returnScriptCode = mgr.lookupBean(ID_ENDMODALSCRIPT);

		synchronized(getEngineLock()) {
			
				// By executing a return script here it still has access to all registered id-beans
				// and will be evaluated in the context of the driver and stored as an InputForm String
				// so when returned any Removed symbols end up in the returning context
			try {
				if (returnScriptCode != null) {
					if (returnScriptCode instanceof String) {
						result = eval(ENDMODAL_SOURCE, -1, -1, (String)returnScriptCode);
						}
					else if (returnScriptCode instanceof MathematicaEventProcessor) {
						result = ((MathematicaEventProcessor)returnScriptCode).process(ENDMODAL_SOURCE);
						}
					mgr.unregisterBean(ID_ENDMODALSCRIPT);
					}
        else
          eval(ENDMODAL_SOURCE, -1, -1, "Null");
				}
			catch (BSFException be) {
				be.printStackTrace();
				}
			}
    return result;
		}
	
	public void processEndModal(Object result) throws MathLinkException {
		
		synchronized(getEngineLock()) {
			KernelLink ml = StdLink.getLink();
			StdLink.requestTransaction();
			synchronized (ml) {
				try {
					ml.putFunction("EvaluatePacket", 1);
					ml.putFunction("CompoundExpression", ((result != null || useEndModalSymbol) ? 2 : 1));
					  ml.putFunction("EndModal", 0);
	          if (useEndModalSymbol) {
	            ml.putSymbol(ENDMODAL_RESULT_STORAGE);
	            }
	          else {
	            if (result != null) {
	              if (result instanceof Expr) {
	                ml.put((Expr)result);
	                }
	              else {
                  ml.putFunction("JavaBlock", 1);
	                ml.putFunction("JavaObjectToExpression", 1);
	                ml.putReference(result);
	                }
	              }
	            }
					ml.discardAnswer();
					} 
				catch (MathLinkException ee) {
					ml.clearError();
					ml.newPacket();
					}
				}

      if (result != null && result instanceof Expr)
        ((Expr)result).dispose();
			}
		
	}
  
  public static KernelLink createKernelLink(String cmdLine, String version) throws MathLinkException {
      KernelLink ml = null;
      String useCmdLine = "-linkmode launch -linkname 'math -mathlink'";

      if (cmdLine == null) {
        // Allows for a command line paramter setting of -DID_KERNELLINKCOMMANDLINE=...
        try {
          cmdLine = System.getProperty(ID_KERNELLINKCOMMANDLINE);
          }
        catch (Exception e){}
        }
      
      if (cmdLine == null) {
        if (Utils.isWindows())
          useCmdLine = "-linkmode launch -linkname 'C:\\Program Files\\Wolfram Research\\Mathematica\\" + 
            (version == null ? DEFAULT_NEWKERNEL_VERSION : version) + "\\MathKernel.exe'";
        else if (Utils.isMacOSX())
          useCmdLine = "-linkmode launch -linkname '\"/Applications/Mathematica " + 
            (version == null ? DEFAULT_NEWKERNEL_VERSION : version) + ".app/Contents/MacOS/MathKernel\" -mathlink'";
        }
      else
        useCmdLine = cmdLine;

      ml = MathLinkFactory.createKernelLink(useCmdLine);

      ml.connect(connectTimeout);

      ml.evaluate("$Line");
      while (ml.waitForAnswer() != MathLink.RETURNPKT)
        ml.newPacket();
      ml.newPacket();

      // NOTE: This currently guarantees the loading of JLink and a call to InstallJava[]
      synchronized (ml) {
        ml.enableObjectReferences();
        }
		
    return ml;
    }

  private void initLink() throws BSFException {
     boolean createdContextName = false;

     if (mathMgr != null && mathMgr.getDebug())
      mathMgr.getDebugStream().println("calling setLink");

    try {

     synchronized(getEngineLock()) {
				KernelLink ml = StdLink.getLink();
				
      // Decide when is a good time to attach any packet listeners

      // For JLink`

			StdLink.requestTransaction();
      synchronized (ml) {
        ml.evaluate("Needs[\"" + KernelLink.PACKAGE_CONTEXT + "\"]");
        ml.discardAnswer();
        }

      StdLink.requestTransaction();
      synchronized (ml) {
        ml.evaluate( "$VersionNumber");
        while (ml.waitForAnswer() != MathLink.RETURNPKT) {
          ml.newPacket();
          }
        mathematicaVersionNumber = new Double(ml.getDouble());
        }

      // Decide when this is best to setup this as active link bean
      // When do we even use this or maybe we only register it if created link??
      mgr.registerBean(ID_KERNELLINK, ml);
      
      Object contextObject = mgr.lookupBean(ID_CONTEXT);

      if (contextObject != null && (contextObject instanceof String)) {
        contextName = (String)contextObject;
        }

      StdLink.requestTransaction();
      synchronized (ml) {
        externalContextName = ml.evaluateToOutputForm("$Context", -1);
        }
          
      // If a context is not provided by the manager then this class
      // generates a new unique one but a relative `Private based context given current $Context
      if (contextName == null) {
			  StdLink.requestTransaction();
        synchronized (ml) {
          contextCount++;
          contextName = ml.evaluateToOutputForm("Begin[\"" + CONTEXT_SUFFIX + contextCount + "`\"]", -1);
          createdContextName = true;
          }
        }

      if (createdContextName) {
			  StdLink.requestTransaction();
        synchronized (ml) {
          ml.evaluate("End[]");
          ml.discardAnswer();
          }
        }

      StdLink.requestTransaction();
        synchronized (ml) {
          ml.evaluate("Begin[\"" + PRIVATE_MATHEMATICABSF_CONTEXT + "\"];");
          ml.discardAnswer();
          }
          
      StringBuffer mathCode = new StringBuffer();

      mathCode.append('(');
            
      mathCode.append('\n');
      
      mathCode.append(SCRIPTEVAL_FUNC);
	  mathCode.append("[ contextName_, scriptPath_, str_] := ( \n");
      mathCode.append("  If[ !StringMatchQ[$Context, contextName],\n");
      mathCode.append("    ToExpression[contextName <> \"" + SCRIPTEXTERNALCONTEXTPATH_SYMBOL + " = $ContextPath\"]];\n");
	  mathCode.append("  Begin[contextName];\n");
      mathCode.append("  $ContextPath = Symbol[contextName <> \"" + SCRIPTCONTEXTPATH_SYMBOL + "\"];\n");
      mathCode.append("  If[!MemberQ[$ContextPath, contextName], $ContextPath = Prepend[$ContextPath, contextName]];\n");
	  mathCode.append("  If[ scriptPath =!= Null,\n");
	  mathCode.append("    $Path = Prepend[$Path, scriptPath]];\n");
	  mathCode.append("  Result = AbortProtect[\n");
	  mathCode.append("     JavaBlock[ToExpression[ \"(\" <> str <> \")\"]]\n");
	  mathCode.append("     ];\n");
	  mathCode.append("  If[ scriptPath =!= Null,\n");
	  mathCode.append("    $Path = DeleteCases[$Path, scriptPath, 1, 1]]; \n");
      mathCode.append("  ToExpression[contextName <> \"" + SCRIPTCONTEXTPATH_SYMBOL + " = $ContextPath\"];\n");
      mathCode.append("  $ContextPath = Symbol[contextName <> \"" + SCRIPTEXTERNALCONTEXTPATH_SYMBOL + "\"];\n");
	  mathCode.append("  End[];\n");
	  mathCode.append("  Result\n");
	  mathCode.append(");\n");
			
      mathCode.append(SCRIPTFILEEVAL_FUNC);
      mathCode.append("[ contextName_, scriptPath_, f_] := ( \n");
      mathCode.append("  If[ !StringMatchQ[$Context, contextName],\n");
      mathCode.append("    ToExpression[contextName <> \"" + SCRIPTEXTERNALCONTEXTPATH_SYMBOL + " = $ContextPath\"]];\n");
      mathCode.append("  Begin[contextName];\n");
      mathCode.append("  $ContextPath = Symbol[contextName <> \"" + SCRIPTCONTEXTPATH_SYMBOL + "\"];\n");
      mathCode.append("  If[!MemberQ[$ContextPath, contextName], $ContextPath = Prepend[$ContextPath, contextName]];\n");
      mathCode.append("  If[ scriptPath =!= Null,\n");
      mathCode.append("    $Path = Prepend[$Path, scriptPath]];\n");
      mathCode.append("  Result = AbortProtect[\n");
      mathCode.append("     JavaBlock[Get[f]]\n");
      mathCode.append("     ];\n");
      mathCode.append("  If[ scriptPath =!= Null,\n");
      mathCode.append("    $Path = DeleteCases[$Path, scriptPath, 1, 1]]; \n");
      mathCode.append("  ToExpression[contextName <> \"" + SCRIPTCONTEXTPATH_SYMBOL + " = $ContextPath\"];\n");
      mathCode.append("  $ContextPath = Symbol[contextName <> \"" + SCRIPTEXTERNALCONTEXTPATH_SYMBOL + "\"];\n");
      mathCode.append("  End[];\n");
      mathCode.append("  Result\n");
      mathCode.append(");\n");
      
	  mathCode.append(SCRIPTEVAL_ENDMODAL_FUNC);
	  mathCode.append("[ contextName_, scriptPath_, str_] := ( \n");
      mathCode.append("  If[ !StringMatchQ[$Context, contextName],\n");
      mathCode.append("    ToExpression[contextName <> \"" + SCRIPTEXTERNALCONTEXTPATH_SYMBOL + " = $ContextPath\"]];\n");
	  mathCode.append("  Begin[contextName];\n");
      mathCode.append("  $ContextPath = Symbol[contextName <> \"" + SCRIPTCONTEXTPATH_SYMBOL + "\"];\n");
      mathCode.append("  If[!MemberQ[$ContextPath, contextName], $ContextPath = Prepend[$ContextPath, contextName]];\n");
	  mathCode.append("  If[ scriptPath =!= Null,\n");
	  mathCode.append("    $Path = Prepend[$Path, scriptPath]];\n");
	  mathCode.append("  " + ENDMODAL_RESULT_STORAGE + " = AbortProtect[\n");
	  mathCode.append("     JavaBlock[ToExpression[ \"(\" <> str <> \")\"]]\n");
	  mathCode.append("     ];\n");
	  mathCode.append("  If[ scriptPath =!= Null,\n");
	  mathCode.append("    $Path = DeleteCases[$Path, scriptPath, 1, 1]];\n");
      mathCode.append("  ToExpression[contextName <> \"" + SCRIPTCONTEXTPATH_SYMBOL + " = $ContextPath\"];\n");
      mathCode.append("  $ContextPath = Symbol[contextName <> \"" + SCRIPTEXTERNALCONTEXTPATH_SYMBOL + "\"];\n");
	  mathCode.append("  End[];\n");
	  mathCode.append("  MatchQ[" + ENDMODAL_RESULT_STORAGE + ", \n");
	  mathCode.append("    _Graphics | _Graphics3D | _GraphicsArray | {__Graphics}]\n");
	  mathCode.append(");\n");
			
      mathCode.append(SCRIPTFILEEVAL_ENDMODAL_FUNC);
      mathCode.append("[ contextName_, scriptPath_, f_] := ( \n");
      mathCode.append("  If[ !StringMatchQ[$Context, contextName],\n");
      mathCode.append("    ToExpression[contextName <> \"" + SCRIPTEXTERNALCONTEXTPATH_SYMBOL + " = $ContextPath\"]];\n");
      mathCode.append("  Begin[contextName];\n");
      mathCode.append("  $ContextPath = Symbol[contextName <> \"" + SCRIPTCONTEXTPATH_SYMBOL + "\"];\n");
      mathCode.append("  If[!MemberQ[$ContextPath, contextName], $ContextPath = Prepend[$ContextPath, contextName]];\n");
      mathCode.append("  If[ scriptPath =!= Null,\n");
      mathCode.append("    $Path = Prepend[$Path, scriptPath]];\n");
      mathCode.append("  " + ENDMODAL_RESULT_STORAGE + " = AbortProtect[\n");
      mathCode.append("     JavaBlock[Get[f]]\n");
      mathCode.append("     ];\n");
      mathCode.append("  If[ scriptPath =!= Null,\n");
      mathCode.append("    $Path = DeleteCases[$Path, scriptPath, 1, 1]];\n");
      mathCode.append("  ToExpression[contextName <> \"" + SCRIPTCONTEXTPATH_SYMBOL + " = $ContextPath\"];\n");
      mathCode.append("  $ContextPath = Symbol[contextName <> \"" + SCRIPTEXTERNALCONTEXTPATH_SYMBOL + "\"];\n");
      mathCode.append("  End[];\n");
      mathCode.append("  MatchQ[" + ENDMODAL_RESULT_STORAGE + ", \n");
      mathCode.append("    _Graphics | _Graphics3D | _GraphicsArray | {__Graphics}]\n");
      mathCode.append(");\n");
    
			mathCode.append(UNDECLARESYMS_FUNC);
			mathCode.append("[patt_, {str___String}] := ( \n");
			mathCode.append("  ReleaseJavaObject[ToExpression[#]]& /@ {str};\n");
			mathCode.append("  ClearAll /@ {str};\n");
			mathCode.append("  Remove /@ {str};\n");
			mathCode.append("  ClearAll[patt];\n");
			mathCode.append("  Remove[patt];\n");
			mathCode.append(");\n");
			
			mathCode.append(EXPRTOXMLSTRING_FUNC);
      // When creating the GUIKit expression is it always the case that 
      // no other contexts/symbols should be expected to evaluate when converting filesystem
      // GUIKit definitions to XML?
			mathCode.append("[str_, format_, contextName_, patt_] := Block[{$ContextPath = {\"GUIKit`\", \"JLink`\", \"System`\"}}, \n");
			mathCode.append("  Begin[contextName];\n");
			mathCode.append("  ToExpression[contextName <> \"PrivateEvaluate\"];\n");
			mathCode.append("  Result = ExportString[ ToExpression[\"(\" <> str <> \")\"], format,  \"ElementFormatting\"->None];\n");
			mathCode.append("  End[];\n");
			mathCode.append("  Clear[patt];\n");
			mathCode.append("  Remove[patt];\n");
			mathCode.append("  Result\n");
			mathCode.append("];\n");
        
      mathCode.append(EXPRFILETOXMLSTRING_FUNC);
      // When creating the GUIKit expression is it always the case that 
      // no other contexts/symbols should be expected to evaluate when converting filesystem
      // GUIKit definitions to XML?
      mathCode.append("[file_, format_, contextName_, patt_] := Block[{$ContextPath = {\"GUIKit`\", \"JLink`\", \"System`\"}}, \n");
      mathCode.append("  Begin[contextName];\n");
      mathCode.append("  ToExpression[contextName <> \"PrivateEvaluate\"];\n");
      mathCode.append("  Result = ExportString[ Get[file], format,  \"ElementFormatting\"->None];\n");
      mathCode.append("  End[];\n");
      mathCode.append("  Clear[patt];\n");
      mathCode.append("  Remove[patt];\n");
      mathCode.append("  Result\n");
      mathCode.append("];\n");
      
			mathCode.append('\n');
     
			mathCode.append(')');

     // Send mathCode to kernel
      StdLink.requestTransaction();
      synchronized (ml) {
        ml.evaluate( mathCode.toString() );
        ml.discardAnswer();
        }

      StdLink.requestTransaction();
        synchronized (ml) {
          ml.evaluate("End[];");
          ml.discardAnswer();
          }
          
			// Any code we evaluate here in setup code should be wrapped with a Begin End of the
			// context so all new symbols end up in this context
			
      StdLink.requestTransaction();
      synchronized (ml) {
        ml.evaluate( "Begin[\"" + contextName + "\"]");
        ml.discardAnswer();
        }

			Object driver = mgr.lookupBean(ID_DRIVER);
			if (driver != null) 
      	declareSymbol(contextName + DRIVER_SYMBOL, driver);

			declareSymbol(contextName + ENGINE_SYMBOL, this);
      declareSymbol(contextName + BSF_SYMBOL, functions);

      // We declare these Mathematica rules here so that all kernel instances
      // can use the BSF functionality without requiring a separate Mathematica package
      // installation

      mathCode = new StringBuffer();

      mathCode.append('(');
      mathCode.append('\n');

      // Standalone BSFEngine uses the Bean syntax and calls directly
      // to the BSFFunctions Java object.

      mathCode.append(
        contextName + "$ScriptContext = \"" + contextName + "\";"
        );

      mathCode.append('\n');
      
      mathCode.append(
        contextName + "$ScriptExternalContext = \"" + externalContextName + "\";"
        );
      
      mathCode.append(
			  contextName + SCRIPTCONTEXTPATH_SYMBOL + " = Join[{\"" + contextName + "\", \"GUIKit`\", \"JLink`\", \"System`\"}, {");
      
      Object includedScriptContexts = mgr.lookupBean(ID_INCLUDEDSCRIPTCONTEXTS);
      if (includedScriptContexts != null) {
        String[] contexts = (String[])includedScriptContexts;
        for (int i = 0; i < contexts.length; ++i) {
          mathCode.append("\"");
          mathCode.append(contexts[i]);
          mathCode.append("\"");
          if (i != contexts.length - 1) mathCode.append(",");
          }
        }
      mathCode.append("}];");

      mathCode.append('\n');
      
      mathCode.append(
        contextName + "ToScriptExternalContext[expr_ ] := ReplaceAll[expr, (s_Symbol /; Context[s] === " +
          contextName +  "$ScriptContext) :> Symbol[" + 
          contextName + "$ScriptExternalContext <> SymbolName[Unevaluated[s]]]];"
         );
      mathCode.append('\n');
      
      mathCode.append(
        contextName + BEAN_SYMBOL + "[str_String] := " +
            contextName + BSF_SYMBOL + " @ lookupBean[str];"
        );

      mathCode.append('\n');

      mathCode.append(
        contextName + BEAN_SYMBOL + "[l_List] := " + contextName + BEAN_SYMBOL +" /@ l;"
        );

      mathCode.append('\n');

      mathCode.append(
        contextName + "Set" + BEAN_SYMBOL + "[(f:(Rule | RuleDelayed))[str_String, obj_]] := " +
            contextName + BSF_SYMBOL + " @ registerBean[str, obj];"
        );

      mathCode.append('\n');

      mathCode.append(
        contextName + "Set" + BEAN_SYMBOL + "[l__] := " +
          contextName + "Set" + BEAN_SYMBOL + " /@ Flatten[{l}];"
        );

      mathCode.append('\n');

      mathCode.append(
        contextName + "Unset" + BEAN_SYMBOL + "[str_String] := " +
            contextName + BSF_SYMBOL + " @ unregisterBean[str];"
        );

      mathCode.append('\n');

      mathCode.append(
        contextName + "Unset" + BEAN_SYMBOL + "[l_List] := " +
          contextName + "Unset" + BEAN_SYMBOL + " /@ l;"
        );
   
     mathCode.append('\n');

     mathCode.append(')');

      // Send mathCode to kernel
			StdLink.requestTransaction();
      synchronized (ml) {
        ml.evaluate( mathCode.toString() );
        ml.discardAnswer();
        }

    StdLink.requestTransaction();
    synchronized (ml) {
      ml.evaluate( "End[]");
      ml.discardAnswer();
      }
        
    // Allow for additional initializations from custom BSFManagers
    if (mathMgr != null) {
      mathMgr.initializeLink(contextName);
      }
 
			}

      }
    catch (MathLinkException e) {
      // How should we cleanup/fail with initialize, probably rethrow BSFException
      if (mathMgr != null && mathMgr.getDebug())
        e.printStackTrace(mathMgr.getDebugStream());

      throw new BSFException(BSFException.REASON_OTHER_ERROR, 
			  (e.getMessage() != null) ? e.getMessage() : e.getClass().getName(), e);
      }

    }


  /**
   * initialize the engine.
   */
  public void initialize(final BSFManager mgr, String lang, Vector declaredBeans) throws BSFException {
    super.initialize (mgr, lang, declaredBeans);

    if (mgr instanceof MathematicaBSFManager)
      mathMgr = (MathematicaBSFManager)mgr;
      
    if (mathMgr != null && mathMgr.getDebug())
      mathMgr.getDebugStream().println("beginning MathematicaBSFEngine initialize");

    functions = new MathematicaBSFFunctions(mgr);

    // We add this so a user can do InvokeMethod["ScriptEvaluator", "abort"]
    // to interrupt engine evaluations.
    functions.registerBean(ID_SCRIPTEVALUATOR, functions);
      
    Object linkObject = mgr.lookupBean(ID_KERNELLINK);
    Object linkCommandLine = mgr.lookupBean(ID_KERNELLINKCOMMANDLINE);
    Object linkVersion = mgr.lookupBean(ID_KERNELLINKVERSION);
		Object useSingleKernel = mgr.lookupBean(ID_USESINGLEKERNELLINK);
		boolean useCreatedKernel = false;
		if ((useSingleKernel != null && (useSingleKernel instanceof Boolean) && ((Boolean)useSingleKernel).booleanValue()))
			useCreatedKernel = true;
			
    if (linkObject != null && (linkObject instanceof KernelLink)) {
			startedAsStdLink = false;
			StdLink.setLink((KernelLink)linkObject);
      initLink();
      }
    else {
      
      try {
        if ((singleKernelLinkCount == 0) && !useCreatedKernel && StdLink.getLink() != null) {
          initLink();
          return;
          }
        }
      catch (Exception ex) {}
      
			startedAsStdLink = false;
      engineCreatedLink = true;
      // By default a single kernel is shared for all engines in one VM

      if (singleKernelLink == null) {
        try {
          singleKernelLink = createKernelLink((String)linkCommandLine, (String)linkVersion);
					StdLink.setLink(singleKernelLink);
          }
        catch (MathLinkException e) {
          throw new BSFException(BSFException.REASON_OTHER_ERROR,
					(e.getMessage() != null) ? e.getMessage() : e.getClass().getName(), e);
          }
        }

      if (singleKernelLink != null) {
				initLink();
        singleKernelLinkCount++;
        }

      }

    }

  public void cleanupContext() {
    if (contextName == null) return;
    if (mathMgr != null && mathMgr.getDebug()) {
			mathMgr.getDebugStream().println("cleaning up context :" + contextName);
			}
        
    synchronized(getEngineLock()) {
			KernelLink ml = StdLink.getLink();
			
			try {
				StdLink.requestTransaction();
				synchronized (ml) {
					ml.putFunction("EvaluatePacket", 1);
					ml.putFunction(UNDECLARESYMS_FUNC, 2);
						ml.put(contextName + "*");
						ml.putFunction("List", declaredSymbols.size());
						Iterator it = declaredSymbols.iterator();
						while(it.hasNext()) {
							ml.put((String)it.next());
							}
					ml.endPacket();
					ml.flush();
					ml.discardAnswer();
					} 

      	}
    	catch (MathLinkException e) {
				// Not guaranteed to be a complete or useful cleanup.
				// TODO should we rethrow as BSFException always?
				if (mathMgr != null && mathMgr.getDebug())
					e.printStackTrace(mathMgr.getDebugStream());

				ml.clearError();
				ml.newPacket();
      	}
  
	  	declaredSymbols.clear();
    	// we set contextName to null to know we have cleaned up
    	// so things like Remove[] are not called again in a terminate cleanup
    	contextName = null;
			}
	
    }

  public void terminate() {

    if (mathMgr != null && mathMgr.getDebug()) {
        mathMgr.getDebugStream().println("Terminate called on Mathematica engine instance");
        }

    synchronized(getEngineLock()) {
			KernelLink ml = StdLink.getLink();
			
    if (engineCreatedLink) {
      if (ml.equals(singleKernelLink) && singleKernelLink != null) {
        singleKernelLinkCount--;
        if (singleKernelLinkCount <= 0) {
          singleKernelLink = null;
          if (mathMgr != null && mathMgr.getDebug()) {
            mathMgr.getDebugStream().println("Terminating KernelLink...");
            }
          ml.terminateKernel();
          ml.close();
          StdLink.setLink(null);
          }
        else {
          cleanupContext();
          // Now unshareKernel since we are done with it, could be a no-op
          unshareKernel();
          }
        }
      else {
        if (mathMgr != null && mathMgr.getDebug()) {
          mathMgr.getDebugStream().println("Terminating KernelLink...");
          }
        ml.terminateKernel();
        ml.close();
        StdLink.setLink(null);
        }
      }
    else {
      cleanupContext();
      // Now unshareKernel since we are done with it, could be a no-op
      unshareKernel();
      }
		}
		
    super.terminate();
    }

  private String resolveToMathematicaPath(URL contextURL) {
    if (contextURL == null) return null;
    // Currently we only support adding to $Path for file:// URLs
    if (!contextURL.getProtocol().equals("file")) return null;
    return contextURL.getFile();
    }

  /**
   * This is used by an application to evaluate a string containing
   * some expression.
   */
  public Object eval(String source, int lineNo, int columnNo, Object oscript) throws BSFException {
    Object retval = null;

    if (contextName == null) return null;
 
      if (mathMgr != null && mathMgr.getDebug()) {
        mathMgr.getDebugStream().println("  engine context : " + contextName);
        mathMgr.getDebugStream().print("  eval : " + oscript.toString());
        mathMgr.getDebugStream().println("");
        }

      synchronized(getEngineLock()) {
			  KernelLink ml = StdLink.getLink();
			  
      try {

        // Note this one gets added and removed only during an eval and call
        //   might want a different general packet listener added that
        // will report non-eval messages/prints??
        // TODO consider making this a simple display/enable
        //  if we know that it is already added as a listener
        if (engineCreatedLink || (mathMgr != null && mathMgr.getDebug())) {
        	if (evalPacketListener == null) {
          	evalPacketListener = new EvalPacketListener(
              (mathMgr != null ? mathMgr.getDebugStream() : System.err));
          	}
        	evalPacketListener.reset();
        	ml.addPacketListener(evalPacketListener);
        	}

        String scriptPath = null;
        Object contextURL = mgr.lookupBean(ID_SCRIPTCONTEXTURL);
        if (contextURL != null && contextURL instanceof URL) {
          if (mathMgr != null && mathMgr.getDebug()) {
            mathMgr.getDebugStream().println("  eval in context URL: " + ((URL)contextURL).toExternalForm());
            }
          scriptPath = resolveToMathematicaPath( (URL)contextURL);
          }
            
        boolean asFile = (oscript instanceof URL);
        
        StdLink.requestTransaction();
        synchronized (ml) {
          ml.putFunction("EvaluatePacket", 1);
            if (ENDMODAL_SOURCE.equals(source)) {
              ml.putFunction( asFile ? SCRIPTFILEEVAL_ENDMODAL_FUNC : SCRIPTEVAL_ENDMODAL_FUNC, 3);
              }
            else ml.putFunction( asFile ? SCRIPTFILEEVAL_FUNC : SCRIPTEVAL_FUNC, 3);
              ml.put(contextName);
              if (scriptPath != null) {
                ml.putFunction("DirectoryName", 1);
                  ml.put(scriptPath);
                }
              else ml.putSymbol("Null");
              if (asFile) ml.put(((URL)oscript).getPath());
              else ml.put(oscript.toString());
          ml.endPacket();
          ml.flush();
 
          retval = getAndWaitForJavaObject();
          }  
            
        if (ENDMODAL_SOURCE.equals(source)) {
          if (retval != null && retval instanceof Boolean && ((Boolean)retval).booleanValue()) {
            useEndModalSymbol = true;
            retval = null;
            }
          else {
            useEndModalSymbol = false;
            StdLink.requestTransaction();
            synchronized (ml) {
              ml.evaluate( "Begin[\"" + contextName + "\"]");
              ml.discardAnswer();
              }
            StdLink.requestTransaction();
            synchronized (ml) {
              ml.putFunction("EvaluatePacket", 1);
                ml.putSymbol(ENDMODAL_RESULT_STORAGE);
              ml.endPacket();
              ml.flush();
              retval = getAndWaitForJavaObject();
              }
            StdLink.requestTransaction();
            synchronized (ml) {
              ml.evaluate( "End[]");
              ml.discardAnswer();
              }  
            }
          }     

        }
     catch (MathLinkException e) {
        if (!ml.clearError() || e.getErrCode() == 11)
          // error 11 is "other side closed the link"
           throw new BSFException(BSFException.REASON_OTHER_ERROR, 
			       (e.getMessage() != null) ? e.getMessage() : e.getClass().getName(), e);
        else {
           if (mathMgr != null && mathMgr.getDebug())
              e.printStackTrace(mathMgr.getDebugStream());
          ml.newPacket();
          }
        }
     finally {
        // If evals can be nested?? We need a counter or stack to know
        // when to remove and to add
        if (engineCreatedLink || (mathMgr != null && mathMgr.getDebug())) {
        	ml.removePacketListener(evalPacketListener);
        	}
        }

		}
    // Will probably want to return this Expr as native Java objects
    return retval;
    }

  // TODO we need to see if it can get expensive if we always build an Expr
  // out of a result or see if we can determine if an eval will need the result or not
  // from a caller
  private Object getAndWaitForJavaObject() throws MathLinkException {
  	KernelLink ml = StdLink.getLink();
  	
    int pkt = ml.waitForAnswer();

    while (pkt != KernelLink.RETURNPKT && pkt != KernelLink.INPUTNAMEPKT) {
			ml.newPacket();
			pkt = ml.waitForAnswer();
		  }

    Object resultObj = null;

    long mk = ml.createMark();
    try {
    int type = ml.getNext();
    switch (type) {
      case KernelLink.MLTKOBJECT:
        resultObj = ml.getObject();
        break;
      case KernelLink.MLTKINT:
        resultObj = new Integer(ml.getInteger());
        break;
      case KernelLink.MLTKREAL:
        resultObj = new Double(ml.getDouble());
        break;
      case KernelLink.MLTKSTR:
        resultObj = ml.getString();
        break;
      case KernelLink.MLTKSYM:
        String sym = ml.getSymbol();
        if ("Null".equals(sym))
          resultObj = null;
        else if ("True".equals(sym))
			    resultObj = Boolean.TRUE;
			  else if ("False".equals(sym))
			    resultObj = Boolean.FALSE;
			  else
          resultObj = new Expr(Expr.SYMBOL, sym);
      	break;
			case KernelLink.MLTKFUNC:
        ml.seekMark(mk);
			  resultObj = ml.getExpr();
        break;
        }
      }
    finally {
      ml.destroyMark(mk);
      }
    return resultObj;
    }


	// NOTE call should not be getting any calls from Inteface`
	// This is an optional BSF engine thing
	
  /**
   * Return an object from an extension.
   * @param object Object on which to make the call (ignored).
   * @param method The name of the method to call.
   * @param args an array of arguments to be
   * passed to the extension, which may be either
   * Vectors of Nodes, or Strings.
   */
  public Object call(Object object, String method, Object[] args) throws BSFException {
    Object theReturnValue = null;
		if (contextName == null) return null;
    if (mathMgr != null && mathMgr.getDebug())
      mathMgr.getDebugStream().println("  call : " + method);

    synchronized(getEngineLock()) {
    	KernelLink ml = StdLink.getLink();
			
    try {

      // Note this one gets added and removed only during an eval and call
      //   might want a different general packet listener added that
      // will report non-eval messages/prints??
      if (engineCreatedLink || (mathMgr != null && mathMgr.getDebug())) {
      	if (evalPacketListener == null) {
        	evalPacketListener = new EvalPacketListener(
            (mathMgr != null ? mathMgr.getDebugStream() : System.err));
        	}
				evalPacketListener.reset();
				ml.addPacketListener(evalPacketListener);
      	}
      
      StdLink.requestTransaction();
      synchronized (ml) {
        ml.evaluate( "Begin[\"" + contextName + "\"]");
        ml.discardAnswer();
        }

      StdLink.requestTransaction();
      synchronized (ml) {
				String scriptPath = null;
				Object contextURL = mgr.lookupBean(ID_SCRIPTCONTEXTURL);
				if (contextURL != null && contextURL instanceof URL) {
					if (mathMgr != null && mathMgr.getDebug()) {
						mathMgr.getDebugStream().println("  eval in context URL: " + ((URL)contextURL).toExternalForm());
						}
					scriptPath = resolveToMathematicaPath( (URL)contextURL);
					}
										
        ml.putFunction("EvaluatePacket", 1);
        // TODO perhaps we wrap with AbortProtect?
        
				if (scriptPath != null) {
					ml.putFunction("Block", 2);
						ml.putFunction("List", 1);
							// $Path = Prepend[$Path, DirectoryName[scriptPath]]
							ml.putFunction("Set", 2);
								ml.putSymbol("$Path");
								ml.putFunction("Prepend", 2);
									ml.putSymbol("$Path");
									ml.putFunction("DirectoryName", 1);
										ml.put(scriptPath);
					}
					
        ml.putFunction(method, args.length);
          for (int i = 0; i < args.length; ++ i)
            ml.put( args[i]);

        ml.endPacket();
        ml.flush();
        theReturnValue = getAndWaitForJavaObject();
        }

      StdLink.requestTransaction();
      synchronized (ml) {
        ml.evaluate( "End[]");
        ml.discardAnswer();
        }

      }
    catch (MathLinkException e) {
      // Not guaranteed to be a complete or useful cleanup.
      // TODO should we rethrow as BSFException always?
      if (mathMgr != null && mathMgr.getDebug())
        e.printStackTrace(mathMgr.getDebugStream());

      ml.clearError();
      ml.newPacket();
      }
    finally {
    	if (engineCreatedLink || (mathMgr != null && mathMgr.getDebug())) {
      	ml.removePacketListener(evalPacketListener);
    		}
      }

		}
	return theReturnValue;
  }

  protected void declareSymbol(String name, Object obj) throws BSFException {
    // TODO we need to verify that the name built up here is a valid Mathematica symbol
    // name, otherwise we throw exception and fail yes?
    if (mathMgr != null && mathMgr.getDebug()) {
      mathMgr.getDebugStream().println("calling declareSymbol for name :" + name);
      mathMgr.getDebugStream().println("  for object :" + obj);
      }
      
    synchronized(getEngineLock()) {
			KernelLink ml = StdLink.getLink();
			
	    try {
				StdLink.requestTransaction();
	      synchronized (ml) {
	        ml.putFunction("EvaluatePacket", 1);
	        // TODO wrap with AbortProtect??
          ml.putFunction("KeepJavaObject", 1);
	          ml.putFunction("Set", 2);
	            ml.putSymbol(name);
	            ml.putReference(obj);
	        ml.endPacket();
	        ml.flush();
	        ml.discardAnswer();
	        }
	      declaredSymbols.add(name);
			  }
	    catch (MathLinkException e) {
				// Not guaranteed to be a complete or useful cleanup.
	      // TODO should we rethrow as BSFException always?
				if (mathMgr != null && mathMgr.getDebug())
	        e.printStackTrace(mathMgr.getDebugStream());
				ml.clearError();
				ml.newPacket();
			  }
			}
		
  }

  public void declareBean(BSFDeclaredBean bean) throws BSFException{
    // Because scripts could use non-letterlikes in bean names
    // we chose to use Bean["name"] for finding a bean
    return;
    /* Not needed because of Bean[""] mappings */
    }
    
  public void undeclareBean(BSFDeclaredBean bean) throws BSFException {
    // Because scripts could use non-letterlikes in bean names
    // we chose to use Bean["name"] for finding a bean
    return;
    }

}
