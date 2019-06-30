/*
 * @(#)InvokeResultRunnable.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.concurrent;

import com.wolfram.bsf.util.type.TypedObject;

import EDU.oswego.cs.dl.util.concurrent.LinkedQueue;

/**
 * InvokeResultRunnable
 */
public abstract class InvokeResultRunnable extends InvokeRunnable {
  
  protected LinkedQueue resultQueue = new LinkedQueue();

	public Object getResult() {
    Object result = null;
    try {
      TypedObject obj = (TypedObject)resultQueue.take();
      if (obj != null) result = obj.value;
      }
    catch (InterruptedException ie) {}
    return result;
    }
    
	public void setResult(Object r) {
    try {
      if (r == null) resultQueue.put(TypedObject.TYPED_NULL);
      else resultQueue.put(new TypedObject(r.getClass(), r));
      }
    catch (InterruptedException ie) {}
    }
  
  public void drainResult() {
    try {
      for (;;) {
        Object item = resultQueue.poll(0);
        if (item == null) break;
        }
      }
    catch(InterruptedException ex) {}
    }
  
  }