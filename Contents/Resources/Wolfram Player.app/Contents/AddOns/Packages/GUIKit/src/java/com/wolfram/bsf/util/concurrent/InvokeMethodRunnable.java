/*
 * @(#)InvokeMethodRunnable.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.concurrent;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * InvokeMethodRunnable
 */
public class InvokeMethodRunnable extends InvokeResultRunnable {
  
    private Method method;
    private Object obj;
    private Object[] args;

    public InvokeMethodRunnable(Method m, Object o, Object[] a) {
      method = m;
      obj = o;
      args = a;
      }
    public void run() {
      Object result = null;
      try {
        result = method.invoke(obj, args);
        }
      catch (IllegalAccessException ie) {
        exception = ie;
        }
      catch (InvocationTargetException ite) {
        exception = ite;
        }
      finally {
        setResult(result);
        }
      
      }
      
    public void cleanup() {
      method = null;
      obj = null;
      args = null;
      }
    
    }