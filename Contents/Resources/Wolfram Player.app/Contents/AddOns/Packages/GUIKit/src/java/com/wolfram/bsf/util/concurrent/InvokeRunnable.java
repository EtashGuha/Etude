/*
 * @(#)InvokeRunnable.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.concurrent;

import java.lang.reflect.InvocationTargetException;

/**
 * InvokeRunnable
 */
public abstract class InvokeRunnable implements Runnable {
  
  protected Exception exception = null;
    
  public abstract void run();

  public abstract void cleanup();
  
  public Exception getException() {return exception;}
  
  public void handleException() throws Exception {
  	if (exception != null) {
			if (exception instanceof IllegalAccessException)
				throw (IllegalAccessException)exception;
			if (exception instanceof InvocationTargetException)
				throw (InvocationTargetException)exception;
  		}
  	}
  
  }