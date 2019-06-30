/*
 * @(#)ThreadQueue.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.concurrent;

import EDU.oswego.cs.dl.util.concurrent.Channel;
import EDU.oswego.cs.dl.util.concurrent.QueuedExecutor;

/**
 * ThreadQueue
 */
public class ThreadQueue extends QueuedExecutor {
   
  /**
   * Construct a new ThreadQueue that uses
   * the supplied Channel as its queue. 
   * <p>
   * This class does not support any methods that 
   * reveal this queue. If you need to access it
   * independently (for example to invoke any
   * special status monitoring operations), you
   * should record a reference to it separately.
   **/
  public ThreadQueue(Channel queue) {
    super(queue);
    }

  /**
   * Construct a new ThreadQueue that uses
   * a BoundedLinkedQueue with the current
   * DefaultChannelCapacity as its queue.
   **/
  public ThreadQueue() {
    super();
    }
  
  /**
   * Drains any Runnables that are currently waiting in the Queue removing
   * them wihtout processing their runs and calling cleanup to null
   * out any potential references
   **/
  public void drain() {
    try {
      for (;;) {
        Object item = queue_.poll(0);
        if (item != null) {
          if (item instanceof InvokeRunnable)
            ((InvokeRunnable)item).cleanup();
          }
        else break;
        }
      }
    catch(InterruptedException ex) {}
    }
  
}
