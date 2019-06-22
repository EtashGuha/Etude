/*
 * @(#)MathematicaEventAdapterProxy.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.event;

import java.beans.EventSetDescriptor;
import java.beans.PropertyChangeEvent;

import java.lang.reflect.*;

import com.wolfram.bsf.util.concurrent.InvokeMode;
import com.wolfram.bsf.util.MathematicaBSFException;
import com.wolfram.bsf.util.MathematicaEngineUtils;
import com.wolfram.bsf.util.MathematicaMethodUtils;
import com.wolfram.bsf.util.type.TypedObject;
import com.wolfram.bsf.util.type.TypedObjectFactory;

/** MathematicaEventAdapterProxy
  *
  * Since we are using the java.lang.reflect.Proxy functionality here
  * this places a minimum requirement of Java VMs versions >= 1.3.x
  **/
public class MathematicaEventAdapterProxy implements InvocationHandler {

	private MathematicaEventProcessor eventprocessor;
  private boolean usePropertyName = false;
  
	// preloaded Method objects for the methods in java.lang.Object
	private static Method hashCodeMethod;
	private static Method equalsMethod;
	private static Method toStringMethod;
		 
	static {
		try {
			 hashCodeMethod = Object.class.getMethod("hashCode", (Class[])null);
			 equalsMethod = Object.class.getMethod("equals", new Class[] { Object.class });
			 toStringMethod = Object.class.getMethod("toString", (Class[])null);
				} 
		catch (NoSuchMethodException e) {
			throw new NoSuchMethodError(e.getMessage());
	 		}
		}
		 
  public static TypedObject newInstance(ClassLoader loader,
      EventSetDescriptor eventsetdescriptor,  MathematicaEventProcessor eventprocessor,
      Object eventedObject ) throws MathematicaBSFException {
        
    boolean usePropName = false;
    
    if(eventsetdescriptor.getName().equals(MathematicaEngineUtils.ATTVAL_PROPERTY_CHANGE) ||
       eventsetdescriptor.getName().equals(MathematicaEngineUtils.ATTVAL_VETOABLE_CHANGE)) {
       usePropName = true;
       }
           
    Object eventadapter = Proxy.newProxyInstance(loader, 
      new Class[]{eventsetdescriptor.getListenerType()},
      new MathematicaEventAdapterProxy(eventprocessor, usePropName));

    Method method = eventsetdescriptor.getAddListenerMethod();
    try {
      MathematicaMethodUtils.callMethod(method, eventedObject,
      	 new Object[]{eventadapter}, new Class[]{eventadapter.getClass()}, InvokeMode.INVOKE_CURRENT);
      }
    catch(Exception exception) {
      Throwable throwable = (exception instanceof InvocationTargetException) ? ((InvocationTargetException)exception).getTargetException() : null;
      throw new MathematicaBSFException(MathematicaBSFException.REASON_OTHER_ERROR, "method invocation failed when calling : " + method.getName() + " : " + exception + (throwable != null ? " target exception: " + throwable : ""), throwable);
      }
    return TypedObjectFactory.create(eventsetdescriptor.getListenerType(), eventadapter);
    }
    
	private MathematicaEventAdapterProxy(MathematicaEventProcessor obj, boolean usePropertyName) {
		this.eventprocessor = obj;
    this.usePropertyName = usePropertyName;
		}
			
	public Object invoke(Object proxy, Method m, Object[] args) throws Throwable {

		if (m.getDeclaringClass() == Object.class) {
			if (m.equals(hashCodeMethod)) {
				return proxyHashCode(proxy);
					} 
			else if (m.equals(equalsMethod)) {
				return proxyEquals(proxy, args[0]);
					} 
			else if (m.equals(toStringMethod)) {
				return proxyToString(proxy);
					}
			else {
				throw new InternalError("unexpected Object method dispatched: " + m);
				}
			}
	  else {
      String eventName = m.getName();
      if(usePropertyName) {
        try {
          String newEventName = ((PropertyChangeEvent)args[0]).getPropertyName();
          // PropertyChangeEvents can have null property names to mean
          // multiple properties have changed
          if (newEventName != null) eventName = newEventName;
          }
        catch (Exception e) {}
        }
         
      // any way we cache this request to eliminate the check each call??
      // the problem is if this is a listener on a multi method listener and
      // some methods throw exceptions and some don't we'd have to create a list 
      // of all possible choices, so let's leave this check here for now
			if (m.getExceptionTypes().length > 0)
				eventprocessor.processExceptionableEvent(eventName, args);
			else
				eventprocessor.processEvent(eventName, args);
	  	}
		return null;
		}

	protected Integer proxyHashCode(Object proxy) {
		return new Integer(System.identityHashCode(proxy));
		}

	protected Boolean proxyEquals(Object proxy, Object other) {
		return (proxy == other ? Boolean.TRUE : Boolean.FALSE);
		}

	protected String proxyToString(Object proxy) {
		return proxy.getClass().getName() + '@' + Integer.toHexString(proxy.hashCode()) + "[eventprocessor=" + eventprocessor.getClass().getName() +
				'@' + Integer.toHexString(eventprocessor.hashCode()) + "]";
		}
			
}