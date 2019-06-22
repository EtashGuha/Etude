package com.wolfram.bsf.engines.javascript;

import java.util.Vector;

import org.mozilla.javascript.Context;
import org.mozilla.javascript.EvaluatorException;
import org.mozilla.javascript.JavaScriptException;
import org.mozilla.javascript.NativeJavaObject;
import org.mozilla.javascript.Scriptable;
import org.mozilla.javascript.ScriptRuntime;
import org.mozilla.javascript.WrappedException;
import org.mozilla.javascript.Wrapper;
import org.mozilla.javascript.ImporterTopLevel;

// BSF import switch
import org.apache.bsf.*;
import org.apache.bsf.util.BSFEngineImpl;
import org.apache.bsf.util.BSFFunctions;
//

import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.bsf.util.MathematicaBSFManager;

/**
 * This is the interface to Netscape's Rhino (JavaScript) from the
 * Bean Scripting Framework.
 * <p>
 * The original version of this code was first written by Adam Peller
 * for use in LotusXSL. Sanjiva took his code and adapted it for BSF.
 *
 * SecuredJavaScriptEngine is a temp replacement for JavaScriptEngine
 * so that we can setup SecuritySupport if someone runs with a js.jar
 * with security turned on (like js.jar from a Batik distribution).
 * In the near future if JavaScriptEngine turns on this feature this
 * class can go away.
 */
public class SecuredJavaScriptEngine extends BSFEngineImpl {

  /**
   * The global script object, where all embedded functions are defined,
   * as well as the standard ECMA "core" objects.
   */
  private Scriptable global;

  private MathematicaSecuritySupport securitySupport;

  /**
   * initialize the engine. put the manager into the context -> manager
   * map hashtable too.
   */
  public void initialize (BSFManager mgr, String lang,
                          Vector declaredBeans) throws BSFException {
    super.initialize (mgr, lang, declaredBeans);

    // Initialize context and global scope object
    try {
      Context cx = Context.enter();
      
			securitySupport = new MathematicaSecuritySupport(
			  (MathematicaBSFEngine)mgr.loadScriptingEngine(MathematicaBSFManager.MATHEMATICA_LANGUAGE_NAME));
			
      cx.setSecuritySupport(securitySupport);
      global = cx.initStandardObjects (new ImporterTopLevel());
      Scriptable bsf = Context.toObject (new BSFFunctions (mgr, this), global);
      global.put ("bsf", global, bsf);
      int size = declaredBeans.size ();
      for (int i = 0; i < size; i++) {
        declareBean ((BSFDeclaredBean) declaredBeans.elementAt (i));
      }
    } finally {
      Context.exit();
    }
  }

  /**
   * This is used by an application to evaluate a string containing
   * some expression.
   */
  public Object eval (String source, int lineNo, int columnNo,
          Object oscript) throws BSFException {
    String script = oscript.toString ();
    Object retval = null;
    try {
      Context cx = Context.enter ();
      cx.setSecuritySupport(securitySupport);
      // Use interpretive mode--generally faster for single executions of scripts.
      cx.setOptimizationLevel(-1);
      retval = cx.evaluateString(global, script, source, lineNo,
          securitySupport.getSecurityDomain(this.getClass()) );
      if (retval instanceof NativeJavaObject)
        retval = ((NativeJavaObject)retval).unwrap();
    } catch (Throwable t) { // includes JavaScriptException, rethrows Errors
      handleError (t);
    } finally {
      Context.exit ();
    }
    return retval;
  }

  /**
   * Return an object from an extension.
   * @param object Object on which to make the call (ignored).
   * @param method The name of the method to call.
   * @param args an array of arguments to be
   * passed to the extension, which may be either
   * Vectors of Nodes, or Strings.
   */
  public Object call (Object object, String method, Object[] args)
                                                        throws BSFException {
    Object theReturnValue = null;

    try {
      Context cx = Context.enter ();
      cx.setSecuritySupport(securitySupport);
      //REMIND: convert arg list Vectors here?

      Object fun = global.get (method, global);
      if (fun == Scriptable.NOT_FOUND) {
	throw new JavaScriptException ("function " + method +
				       " not found.");
      }

      theReturnValue = ScriptRuntime.call (cx, fun, global, args, null);
      if (theReturnValue instanceof Wrapper) {
	theReturnValue = ((Wrapper) theReturnValue).unwrap ();
      }
    } catch (Throwable t) {
      handleError (t);
    } finally {
      Context.exit ();
    }
    return theReturnValue;
  }

  public void declareBean (BSFDeclaredBean bean) throws BSFException {
    // Must wrap non-scriptable objects before presenting to Rhino
    Scriptable wrapped = Context.toObject (bean.bean, global);
    global.put (bean.name, global, wrapped);
  }

  public void undeclareBean (BSFDeclaredBean bean) throws BSFException {
    global.delete (bean.name);
  }

  private void handleError (Throwable t) throws BSFException {
    if (t instanceof WrappedException) {
      t = (Throwable)((WrappedException)t).unwrap();
    }

    String message = null;
    Throwable target = t;

    if (t instanceof JavaScriptException) {
      message = t.getLocalizedMessage();

      // Is it an exception wrapped in a JavaScriptException?
      Object value = ((JavaScriptException)t).getValue();
      if (value instanceof Throwable) {
        // likely a wrapped exception from a LiveConnect call.
        // Display its stack trace as a diagnostic
        target = (Throwable)value;
      }
    } else if (t instanceof EvaluatorException
               || t instanceof SecurityException) {
      message = t.getLocalizedMessage();
    } else if (t instanceof RuntimeException) {
      message = "Internal Error: " + t.toString();
    } else if (t instanceof StackOverflowError) {
      message = "Stack Overflow";
    }

    if (message == null) {
      message = t.toString();
    }

    //REMIND: can we recover the line number here?  I think
    // Rhino does this by looking up the stack for bytecode
    // see Context.getSourcePositionFromStack()
    // but I don't think this would work in interpreted mode

    if (t instanceof Error && !(t instanceof StackOverflowError)) {
      // Re-throw Errors because we're supposed to let the JVM see it
      // Don't re-throw StackOverflows, because we know we've
      // corrected the situation by aborting the loop and
      // a long stacktrace would end up on the user's console
        throw (Error)t;
    } else {
      throw new BSFException(BSFException.REASON_OTHER_ERROR,
			     "JavaScript Error: " + message,
			     target);
    }
  }

  /**
   * Copies and translates any Vector top-level elements of an Object array
   * into Object arrays.
   *
   * @param args incoming Object array; not modified
   * @return Object array with Vectors translated to arrays
   */
  /* This code is specific to LotusXSL and probably isn't needed by BSF
  public Object[] translateArgList(Object[] args)
  {
    //REMIND: construct associative arrays instead of simple arrays,
    // using node name as index as well as integer

    Object[] argList = null;

    if (args != null) {
      argList = new Object[args.length];
      for (int i = 0; i < args.length; i++)
      {
        Object arg = args[i];
        if (arg instanceof Vector)
        {
          // Copy the Vector argument into an Object Array
          Vector vector = (Vector)arg;
          Object[] array = new Object[vector.size()];

          for (int j = 0; j < vector.size(); j++)
          {
            array[j] = vector.elementAt(j);
          }

          arg = array;
        }

        argList[i] = arg;
      }
    }

    return argList;
  }
   */
}
