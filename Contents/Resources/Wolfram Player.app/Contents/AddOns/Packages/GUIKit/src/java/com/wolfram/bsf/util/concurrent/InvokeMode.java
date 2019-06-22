/*
 * @(#)InvokeMode.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.concurrent;

import javax.swing.SwingUtilities;

import EDU.oswego.cs.dl.util.concurrent.Executor;

import com.wolfram.bsf.util.MathematicaBSFManager;

/**
 * InvokeMode
 */
public class InvokeMode  {
  
  public static final String THREAD_CURRENT = "Current";
  public static final String THREAD_DISPATCH = "Dispatch";
  public static final String THREAD_NEW = "New";
  
  public static final int EXECUTE_LATER = 1;
  public static final int EXECUTE_WAIT = 2;
  
  public static final InvokeMode INVOKE_CURRENT = new InvokeMode(THREAD_CURRENT, EXECUTE_WAIT);
  
  private String threadContext = THREAD_CURRENT;
  private int executionMode = EXECUTE_WAIT;
  
  private MathematicaBSFManager manager = null;
  
  // Currently we aren't limiting the number of active new threads
  // but normally there won't be many active Runnable requests
  // to make this grow and it will shutdown created threads
  // with a keepalive setting (currently one minute)
  private static ThreadPool newThreadPool = new ThreadPool();
    
	public InvokeMode() {
		this(THREAD_CURRENT, EXECUTE_WAIT);
		}
	
  public InvokeMode(String context, int mode) {
  	if (context != null) this.threadContext = context;
		if (mode == EXECUTE_WAIT) this.executionMode = EXECUTE_WAIT;
		else this.executionMode = EXECUTE_LATER;
    }
  			
  public static void process(InvokeMode mode, final InvokeRunnable r) throws Exception {
		if (mode.isNewThread()) {
      newThreadPool.execute(r);
			}
		else if (mode.isDispatchThread() && mode.isExecuteLater()) {
			SwingUtilities.invokeLater(r);
			}
		else if (mode.isDispatchThread() && !mode.isExecuteLater()) {
			try {
				SwingUtilities.invokeAndWait(r);
				}
			catch (InterruptedException e) {}
			r.handleException();
			}
    else {
			// Will use newThreadPool if threadContext does not resolve to a valid Executor instance
			Executor exec = newThreadPool;
			if (mode.getManager() != null && mode.getThreadContext() != null) {
				Object obj = mode.getManager().lookupBean( mode.getThreadContext());
				if (obj != null && obj instanceof Executor) exec = (Executor)obj;
				}
			exec.execute(r);
      }
  	}
  
  public static Object processResult(InvokeMode mode, final InvokeResultRunnable r) throws Exception {
		Object result = null;
		if (mode.isNewThread()) {
      newThreadPool.execute(r);
      if (!mode.isExecuteLater()) {
        result = r.getResult();
        }
			}
		else if (mode.isDispatchThread() && mode.isExecuteLater()) {
			SwingUtilities.invokeLater(r);
			}
		else if (mode.isDispatchThread() && !mode.isExecuteLater()) {
			try {
				SwingUtilities.invokeAndWait(r);
				}
			catch (InterruptedException e) {}
			r.handleException();
			result = r.getResult();
			}
    else {
    	// Will use newThreadPool if threadContext does not resolve to a valid Executor instance
    	Executor exec = newThreadPool;
    	if (mode.getManager() != null && mode.getThreadContext() != null) {
    		Object obj = mode.getManager().lookupBean( mode.getThreadContext());
    		if (obj != null && obj instanceof Executor) exec = (Executor)obj;
    		}
			exec.execute(r);
      if (!mode.isExecuteLater()) {
        result = r.getResult();
        }
      }
    r.drainResult();
	  return result;
  	}
  
  public String getThreadContext() {return threadContext;}
  
  public int getExecutionMode() {return executionMode;}
  
  public MathematicaBSFManager getManager() {return manager;}
  public void setManager(MathematicaBSFManager man) {
  	manager = man;
  	}
  
  // Useful utility methods for common cases
  
  public boolean isCurrentThread() {return THREAD_CURRENT.equalsIgnoreCase(threadContext);}
  public boolean isDispatchThread() {return THREAD_DISPATCH.equalsIgnoreCase(threadContext);}
  public boolean isNewThread() {return THREAD_NEW.equalsIgnoreCase(threadContext);}
  public boolean isCustomThreadContext() {return !(isCurrentThread() || isDispatchThread() || isNewThread());}
  
  public boolean isExecuteLater() {return EXECUTE_LATER == executionMode;}
  
  }