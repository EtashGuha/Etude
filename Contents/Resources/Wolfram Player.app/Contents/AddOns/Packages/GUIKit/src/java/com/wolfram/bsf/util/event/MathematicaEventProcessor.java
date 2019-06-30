/*
 * @(#)MathematicaEventProcessor.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.event;

// BSF import switch
import org.apache.bsf.util.event.EventProcessor;
//

/**
 * MathematicaEventProcessor
 */
public interface MathematicaEventProcessor extends EventProcessor {

  public Object process(String evalContext); 

}
