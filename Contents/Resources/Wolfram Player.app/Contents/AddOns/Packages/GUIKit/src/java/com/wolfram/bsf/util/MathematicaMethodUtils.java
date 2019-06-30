/*
 *
 */
package com.wolfram.bsf.util;

// BSF import switch
import org.apache.bsf.util.MethodUtils;
//

import com.wolfram.bsf.util.concurrent.InvokeGetFieldRunnable;
import com.wolfram.bsf.util.concurrent.InvokeMode;
import com.wolfram.bsf.util.concurrent.InvokeFieldRunnable;
import com.wolfram.bsf.util.concurrent.InvokeMethodRunnable;
import com.wolfram.bsf.util.type.MathematicaTypeConvertorRegistry;
import com.wolfram.bsf.util.type.TypedObject;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

import javax.swing.SwingUtilities;

/**
 * <p> Utility reflection methods focussed on methods in general rather than properties in particular. </p>
 *
 * <h3>Known Limitations</h3>
 * <h4>Accessing Public Methods In A Default Access Superclass</h4>
 * <p>There is an issue when invoking public methods contained in a default access superclass.
 * Reflection locates these methods fine and correctly assigns them as public.
 * However, an <code>IllegalAccessException</code> is thrown if the method is invoked.</p>
 *
 * <p><code>MethodUtils</code> contains a workaround for this situation.
 * It will attempt to call <code>setAccessible</code> on this method.
 * If this call succeeds, then the method can be invoked as normal.
 * This call will only succeed when the application has sufficient security privilages.
 * If this call fails then a warning will be logged and the method may fail.</p>
 */

 // Added widening of isAssignmentCompatible and removed dependency on logging
 // Jeff Adams

public class MathematicaMethodUtils {

    // --------------------------------------------------------- Private Methods

    /** An empty class array */
    private static final Class[] emptyClassArray = new Class[0];
    /** An empty object array */
    private static final Object[] emptyObjectArray = new Object[0];

    // --------------------------------------------------------- Public Methods

    /**
     * <p>Invoke a named method whose parameter type matches the object type.</p>
     *
     * It loops through all methods with names that match
     * and then executes the first it finds with compatable parameters.</p>
     *
     * <p>This method supports calls to methods taking primitive parameters
     * via passing in wrapping classes. So, for example, a <code>Boolean</code> class
     * would match a <code>boolean</code> primitive.</p>
     *
     * <p> This is a convenient wrapper for
     * {@link #invokeMethod(Object object,String methodName,Object [] args)}.
     * </p>
     *
     * @param object invoke method on this object
     * @param methodName get method with this name
     * @param arg use this argument
     *
     * @throws NoSuchMethodException if there is no such accessible method
     * @throws InvocationTargetException wraps an exception thrown by the
     *  method invoked
     * @throws IllegalAccessException if the requested method is not accessible
     *  via reflection
     */
    public static Object invokeMethod(TypedObject bean, String methodName, Object arg, 
      InvokeMode mode)
      throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {

    Object[] args = {arg};
    return invokeMethod(bean, methodName, args, mode);
    }
    
  /** Class.getConstructor() finds only the entry point (if any)
    _exactly_ matching the specified argument types. Our implmentation
    can decide between several imperfect matches, using the same
    search algorithm as the Java compiler.

    Note that all constructors are static by definition, so
    isStaticReference is true.

    @exception NoSuchMethodException if constructor not found.
    */
	public static Constructor getConstructor(Class targetClass, Class[] argTypes) throws SecurityException, NoSuchMethodException {
    Constructor c = null;
    try {
      c = MethodUtils.getConstructor(targetClass, argTypes);
      }
    catch (NoSuchMethodException ns) {
      // As a second check see if we can find a constructor with convertable arg types
      if (argTypes != null && argTypes.length > 0) {
        c = getConvertableConstructor(targetClass, argTypes);
        if (c != null) return c;
        }
      throw (ns);
      }
    return c;
  	}
     
  public static Constructor getConvertableConstructor(Class targetClass, Class[] argTypes) throws SecurityException {
    Constructor m = null;

    try {
      return targetClass.getConstructor(argTypes);
      } 
    catch (NoSuchMethodException e) {}

    Constructor[] methods = targetClass.getConstructors();
    if(0 == methods.length) {
      return null;
      }

    // Second pass is to see if a method exists with same arguments and valid convertor classes
    for (int i = 0, size = methods.length; i < size ; i++) {
      Constructor currMethod = methods[i];
 
        // compare parameters
        Class[] methodsParams = currMethod.getParameterTypes();
        int methodParamSize = methodsParams.length;
        if (methodParamSize == argTypes.length) {
          boolean match = true;
          for (int n = 0 ; n < methodParamSize; n++) {
            
            if (!isAssignmentCompatibleOrConvertable(methodsParams[n], argTypes[n])) {
              match = false;
              break;
              }
            }
          if (match) { // get accessible version of method
            if (Modifier.isPublic(currMethod.getModifiers())) {
              return currMethod;
              }
            }
         } 
      }

    return m;
    }
  
  public static Object callGetField(Field field, Object object, InvokeMode mode)
     throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
    Object result = null;
    if (mode.isCurrentThread() || (mode.isDispatchThread() && SwingUtilities.isEventDispatchThread()))
      result = field.get(object);
    else {
      try {
        result = InvokeMode.processResult(mode, new InvokeGetFieldRunnable(field, object));
        }
      catch (Exception e) {
        if (e instanceof IllegalAccessException) throw (IllegalAccessException)e;
        else if (e instanceof InvocationTargetException) throw (InvocationTargetException)e;
        }
      }    
    return result;
    }
    
  public static void callSetField(Field field, Object object, Object val, InvokeMode mode)
     throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
     
    if (mode.isCurrentThread() || (mode.isDispatchThread() && SwingUtilities.isEventDispatchThread()))
      field.set(object, val);
    else {
    	try {
      	InvokeMode.process(mode, new InvokeFieldRunnable(field, object, val));
    		}
    	catch (Exception e) {
    		if (e instanceof IllegalAccessException) throw (IllegalAccessException)e;
    		else if (e instanceof InvocationTargetException) throw (InvocationTargetException)e;
    		}
      }    
    }
     
  public static Object callMethod(Method method, Object object, Object[] args, Class[] argTypes, InvokeMode mode) 
      throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
    Object result = null;
        
    // TODO do we allow for casting conversion/downcasting of
    // different types, most notably double down to float
    // and if we do here is where we might need to convert object types
    // ie. create a useArgs list frm args by using the method's argument types
    //   because previously we must have allowed this method with args call
    //   by looking up that conversion is possible either with a converter
    //   or the casting would work within the invoke
    // if elsewhere we allow double to float conversion automatically or other
    // general object type conversion
    Object[] useArgs = args;
    
    if (argTypes != null) {
      useArgs = new Object[args.length];
      Class[] methodsParams = method.getParameterTypes();
      int methodParamSize = methodsParams.length;
      for (int i = 0; i < methodParamSize; ++i) {
        try{
          useArgs[i] = MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(
            argTypes[i], args[i], methodsParams[i]);
          }
        catch (MathematicaBSFException me) {
          useArgs[i] = args[i];
          }
        }
      }
    
    if (mode.isCurrentThread() || (mode.isDispatchThread() && SwingUtilities.isEventDispatchThread()))
      result = method.invoke(object, useArgs);
    else {
    	try {
      	result = InvokeMode.processResult(mode, new InvokeMethodRunnable(method, object, useArgs));
				}
			catch (Exception e) {
				if (e instanceof IllegalAccessException) throw (IllegalAccessException)e;
				else if (e instanceof InvocationTargetException) throw (InvocationTargetException)e;
				}
      }
    return result;
    }
  
  public static Object invokeMethod(TypedObject bean, String methodName, InvokeMode mode)
    throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {

    Class targetClass = bean.value.getClass();
    if(bean.type == (java.lang.Class.class)) {
      targetClass = (Class)bean.value;
      }
      
    Method method = getMatchingAccessibleMethod(targetClass, methodName, emptyClassArray);
    if (method == null)
      throw new NoSuchMethodException("No such accessible method: " +
        methodName + "(" + createClassParametersString(emptyClassArray) + ") on target class: " + targetClass.getName());      
    return callMethod(method, bean.value, emptyObjectArray, null, mode);
    }

    /**
     * <p>Invoke a named method whose parameter type matches the object type.</p>
     *
     * It loops through all methods with names that match
     * and then executes the first it finds with compatable parameters.</p>
     *
     * <p>This method supports calls to methods taking primitive parameters
     * via passing in wrapping classes. So, for example, a <code>Boolean</code> class
     * would match a <code>boolean</code> primitive.</p>
     *
     * <p> This is a convenient wrapper for
     * {@link #invokeMethod(Object object,String methodName,Object [] args,Class[] parameterTypes)}.
     * </p>
     *
     * @param object invoke method on this object
     * @param methodName get method with this name
     * @param args use these arguments - treat null as empty array
     *
     * @throws NoSuchMethodException if there is no such accessible method
     * @throws InvocationTargetException wraps an exception thrown by the
     *  method invoked
     * @throws IllegalAccessException if the requested method is not accessible
     *  via reflection
     */
    public static Object invokeMethod(TypedObject bean, String methodName,
      Object[] args, InvokeMode mode)
        throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {

      if (args == null) {
        args = emptyObjectArray;
        }
      int arguments = args.length;
      Class parameterTypes [] = new Class[arguments];
      for (int i = 0; i < arguments; i++) {
        if (args[i] == null) parameterTypes[i] = Object.class;
        else parameterTypes[i] = args[i].getClass();
        }
      return invokeMethod(bean, methodName, args, parameterTypes, mode);
      }

    public static String createClassParametersString(Class[] params) {
       if (params == null || params.length == 0) return "";
       StringBuffer buff = new StringBuffer();
       for (int i = 0; i < params.length; ++i) {
        if (i > 0) buff.append(",");
        buff.append(params[i].getName());
        }
       return buff.toString();
      }
      
    /**
     * <p>Invoke a named method whose parameter type matches the object type.</p>
     *
     * It loops through all methods with names that match
     * and then executes the first it finds with compatable parameters.</p>
     *
     * <p>This method supports calls to methods taking primitive parameters
     * via passing in wrapping classes. So, for example, a <code>Boolean</code> class
     * would match a <code>boolean</code> primitive.</p>
     *
     *
     * @param object invoke method on this object
     * @param methodName get method with this name
     * @param args use these arguments - treat null as empty array
     * @param parameterTypes match these parameters - treat null as empty array
     *
     * @throws NoSuchMethodException if there is no such accessible method
     * @throws InvocationTargetException wraps an exception thrown by the
     *  method invoked
     * @throws IllegalAccessException if the requested method is not accessible
     *  via reflection
     */
    public static Object invokeMethod(
      TypedObject bean, String methodName, Object[] args, Class[] parameterTypes, InvokeMode mode)
     throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {

    if (parameterTypes == null) {
      parameterTypes = emptyClassArray;
      }
    if (args == null) {
      args = emptyObjectArray;
      }

    Class targetClass = bean.value.getClass();
    if(bean.type == (java.lang.Class.class)) {
      targetClass = (Class)bean.value;
      }
      
    Method method = getMatchingAccessibleMethod(targetClass, methodName, parameterTypes);
    if (method == null)
      throw new NoSuchMethodException("No such accessible method: " +
         methodName + "("+ createClassParametersString(parameterTypes) + ") on target class: " + targetClass.getName());  
         
    return callMethod(method, bean.value, args, parameterTypes, mode);
    }


    /**
     * <p>Return an accessible method (that is, one that can be invoked via
     * reflection) that implements the specified Method.  If no such method
     * can be found, return <code>null</code>.</p>
     *
     * @param method The method that we wish to call
     */
    public static Method getAccessibleMethod(Method method) {

        // Make sure we have a method to check
        if (method == null) {
            return (null);
        }

        // If the requested method is not public we cannot call it
        if (!Modifier.isPublic(method.getModifiers())) {
            return (null);
        }

        // If the declaring class is public, we are done
        Class clazz = method.getDeclaringClass();
        if (Modifier.isPublic(clazz.getModifiers())) {
            return (method);
        }

        // Check the implemented interfaces and subinterfaces
        method =
                getAccessibleMethodFromInterfaceNest(clazz,
                        method.getName(),
                        method.getParameterTypes());
        return (method);
    }


    // -------------------------------------------------------- Private Methods

	private static boolean arrayContentsEq(Object[] a1, Object[] a2) {
		if (a1 == null) {
      return a2 == null || a2.length == 0;
		 }
		if (a2 == null) {
      return a1.length == 0;
      }
		if (a1.length != a2.length) {
      return false;
		 }
		for (int i = 0; i < a1.length; i++) {
      if (a1[i] != a2[i]) {
        return false;
				}
		 }
    return true;
   }
	
    /**
     * <p>Return an accessible method (that is, one that can be invoked via
     * reflection) that implements the specified method, by scanning through
     * all implemented interfaces and subinterfaces.  If no such method
     * can be found, return <code>null</code>.</p>
     *
     * <p> There isn't any good reason why this method must be private.
     * It is because there doesn't seem any reason why other classes should
     * call this rather than the higher level methods.</p>
     *
     * @param clazz Parent class for the interfaces to be checked
     * @param methodName Method name of the method we wish to call
     * @param parameterTypes The parameter type signatures
     */
    private static Method getAccessibleMethodFromInterfaceNest
            (Class clazz, String methodName, Class parameterTypes[]) {

        Method method = null;
				String lowerMethodName = methodName.toLowerCase();
				
        // Search up the superclass chain
        for (; clazz != null; clazz = clazz.getSuperclass()) {

            // Check the implemented interfaces of the parent class
            Class interfaces[] = clazz.getInterfaces();
            for (int i = 0; i < interfaces.length; i++) {

                // Is this interface public?
                if (!Modifier.isPublic(interfaces[i].getModifiers()))
                    continue;

                // Does the method exist on this interface?
         
                //method = interfaces[i].getDeclaredMethod(methodName, parameterTypes);
                // We need to support case-insensitive matching
                   Method[] methods = interfaces[i].getDeclaredMethods();
                   for (int j = 0; j < methods.length; ++j) {
                   	 if (lowerMethodName.equalsIgnoreCase(methods[i].getName()) &&
										arrayContentsEq(parameterTypes, methods[i].getParameterTypes())
                   	     )
                   	    method = methods[i];
                   	    break;
                   	 }
        
                if (method != null)
                    break;

                // Recursively check our parent interfaces
                method =
                        getAccessibleMethodFromInterfaceNest(interfaces[i],
                                methodName,
                                parameterTypes);
                if (method != null)
                    break;

            }

        }

        // If we found a method return it
        if (method != null)
            return (method);

        // We did not find anything
        return (null);

    }

    /**
     * <p>Find an accessible method that matches the given name and has compatible parameters.
     * Compatible parameters mean that every method parameter is assignable from
     * the given parameters.
     * In other words, it finds a method with the given name
     * that will take the parameters given.<p>
     *
     * <p>This method is slightly undeterminstic since it loops
     * through methods names and return the first matching method.</p>
     *
     * <p>This method is used by
     * {@link
     * #invokeMethod(Object object,String methodName,Object [] args,Class[] parameterTypes)}.
     *
     * <p>This method can match primitive parameter by passing in wrapper classes.
     * For example, a <code>Boolean</code> will match a primitive <code>boolean</code>
     * parameter.
     *
     * @param clazz find method in this class
     * @param methodName find method with this name
     * @param parameterTypes find method with compatible parameters
     */
    public static Method getMatchingAccessibleMethod(
      Class clazz, String methodName, Class[] parameterTypes) {
				String lowerMethodName = methodName.toLowerCase();
        // see if we can find the method directly
        // most of the time this works and it's much faster
        try {
          Method method = clazz.getMethod(lowerMethodName, parameterTypes);
          try {
                // Default access superclass workaround
                // When a public class has a default access superclass
                // with public methods, these methods are accessible.
                // Calling them from compiled code works fine.
                // Unfortunately, using reflection to invoke these methods
                // seems to (wrongly) to prevent access even when the method
                // modifer is public.
                // The following workaround solves the problem but will only
                // work from sufficiently privilages code.
            method.setAccessible(true);
            } 
           catch (SecurityException se) { }
           return method;
        } catch (NoSuchMethodException e) { /* SWALLOW */ }

        // search through all methods
        int paramSize = 0;
        if (parameterTypes != null)
          paramSize = parameterTypes.length;
        Method[] methods = clazz.getMethods();
        
        for (int i = 0, size = methods.length; i < size ; i++) {
          Method currMethod = methods[i];
          if (currMethod.getName().equalsIgnoreCase(lowerMethodName)) {
            // compare parameters
            Class[] methodsParams = currMethod.getParameterTypes();
            int methodParamSize = methodsParams.length;
            if (methodParamSize == paramSize) {
              boolean match = true;
              for (int n = 0 ; n < methodParamSize; n++) {
                if (!isAssignmentCompatible(methodsParams[n], parameterTypes[n])) {
                  match = false;
                  break;
                  }
                }

              if (match) { // get accessible version of method
                Method method = getAccessibleMethod(currMethod);
                if (method != null) {
                  try { // Default access superclass workaround (See above for more details.)
                    method.setAccessible(true);
                    } 
                  catch (SecurityException se) {}
                  return method;
                  }
                }
                
              }
            }
          }
          
         // Second pass is to see if a method exists with same arguments and valid convertor classes
        for (int i = 0, size = methods.length; i < size ; i++) {
          Method currMethod = methods[i];
          if (currMethod.getName().equalsIgnoreCase(lowerMethodName)) {
            // compare parameters
            Class[] methodsParams = currMethod.getParameterTypes();
            int methodParamSize = methodsParams.length;
            if (methodParamSize == paramSize) {
              boolean match = true;
              for (int n = 0 ; n < methodParamSize; n++) {
                
                if (!isAssignmentCompatibleOrConvertable(methodsParams[n], parameterTypes[n])) {
                  match = false;
                  break;
                  }
                }

              if (match) { // get accessible version of method
                Method method = getAccessibleMethod(currMethod);
                if (method != null) {
                  try { // Default access superclass workaround (See above for more details.)
                    method.setAccessible(true);
                    } 
                  catch (SecurityException se) {}
                  return method;
                  }
                }
                
              }
            }
          }
          
        // didn't find a match
        return null;
      }

    /**
     * <p>Determine whether a type can be used as a parameter in a method invocation.
     * This method handles primitive conversions correctly.</p>
     *
     * <p>In order words, it will match a <code>Boolean</code> to a <code>boolean</code>,
     * a <code>Long</code> to a <code>long</code>,
     * a <code>Float</code> to a <code>float</code>,
     * a <code>Integer</code> to a <code>int</code>,
     * and a <code>Double</code> to a <code>double</code>.
     * Now logic widening matches are allowed.
     * For example, a <code>Long</code> will not match a <code>int</code>.
     *
     * @param parameterType the type of parameter accepted by the method
     * @param parameterization the type of parameter being tested
     *
     * @return true if the assignement is compatible.
     */
    protected static final boolean isAssignmentCompatible(Class parameterType, Class parameterization) {
      // try plain assignment
      if (parameterType.isAssignableFrom(parameterization)) {
        return true;
        }

     // TODO
     /* Consider moving to a typeConvertor lookup where we add these and
      *   downsampling convertors and then also make sure places
      * that use this for validation will also go through 
      * and convert the object array of arguments before calling the method
      * Also this opens up matching on more methods so instead of stoping
      * at first matched method name, consider finding all valid method names
      *  with right argument length, then choosing most general or first?
      */
      
      if (parameterType.isPrimitive()) {
        // does anyone know a better strategy than comparing names?
        // also, this method *does* do widening - you must specify exactly
        // byte -> short -> int -> long -> float -> double
        // char -> int -> long -> float -> double
        // is this the right behaviour?
            if (boolean.class.equals(parameterType)) {
              return Boolean.class.equals(parameterization);
              }
            if (char.class.equals(parameterType)) {
              return Character.class.equals(parameterization);
              }
            if (byte.class.equals(parameterType)) {
              return Byte.class.equals(parameterization);
              }
            if (short.class.equals(parameterType)) {
              return Short.class.equals(parameterization) ||
              Byte.class.equals(parameterization);
              }
            if (int.class.equals(parameterType)) {
              return Integer.class.equals(parameterization) ||
                  Short.class.equals(parameterization) ||
                  Byte.class.equals(parameterization) ||
                  Character.class.equals(parameterization);
              }
            if (long.class.equals(parameterType)) {
              return Long.class.equals(parameterization) ||
                  Integer.class.equals(parameterization) ||
                  Short.class.equals(parameterization) ||
                  Byte.class.equals(parameterization) ||
                  Character.class.equals(parameterization);
              }
            if (float.class.equals(parameterType)) {
              return Float.class.equals(parameterization) ||
                  Long.class.equals(parameterization) ||
                  Integer.class.equals(parameterization) ||
                  Short.class.equals(parameterization) ||
                  Byte.class.equals(parameterization) ||
                  Character.class.equals(parameterization);
              }
            if (double.class.equals(parameterType)) {
              return Double.class.equals(parameterization) ||
                  Float.class.equals(parameterization) ||
                  Long.class.equals(parameterization) ||
                  Integer.class.equals(parameterization) ||
                  Short.class.equals(parameterization) ||
                  Byte.class.equals(parameterization) ||
                  Character.class.equals(parameterization);
              }
          }

        // A parameter would be assigned Object.class if null and
        // so we add this acceptance here
        if (Object.class.equals(parameterization)) return true;
        
        return false;
    }
    
  protected static final boolean isAssignmentCompatibleOrConvertable(Class parameterType, Class parameterization) {
      if (isAssignmentCompatible(parameterType, parameterization)) return true;
      if (MathematicaTypeConvertorRegistry.typeConvertorRegistry.lookup(parameterization, parameterType) != null) return true;
      return false;
      }
    
}
