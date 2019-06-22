package com.wolfram.links.rlink;

/**
 * 
 *  RLinkJ source code (c) 2011-2012, Wolfram Research, Inc. 
 *  
 *  
 *  
 *   This file is part of RLinkJ interface to JRI Java library.
 *
 *   RLinkJ is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as 
 *   published by the Free Software Foundation, either version 2 of 
 *   the License, or (at your option) any later version.
 *
 *   RLinkJ is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public 
 *   License along with RLinkJ. If not, see <http://www.gnu.org/licenses/>.
 *  
 * 
 * 
 *
 * @author Leonid Shifrin
 *
 */

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.io.Writer;
import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Map;


import org.apache.log4j.Logger;
import org.rosuda.JRI.Rengine;

import com.wolfram.links.rlink.exceptions.RLinkException;

public class RLinkInit {

	private static RLinkInit init = null;
	private static String[] args = new String[0];
	public static Logger rlogger = Logger.getLogger("RLink");
	private static String nativeLibLocation = null;
	public static String lastError = null;
	private RExecutor exec = null;
	public static final String FUNCTION_HASH_VAR_NAME = "RLinkFunctionHash12345";	
	
	
	public static String getNativeLibLocation() {
		return nativeLibLocation;
	}


	public static void setNativeLibLocation(String nativeLibLocation) {
		RLinkInit.nativeLibLocation = nativeLibLocation;
	}


	/**
	 * 
	 * This function adds a directory to the Java library path. It actually works. Taken from here:
	 * 
	 *   http://stackoverflow.com/questions/5419039/is-djava-library-path-equivalent-to-system-setpropertyjava-library-path
	 * 
	 * 
	 * 
	 * @param s
	 * @throws IOException
	 */
	
	public static void addDir(String s) throws IOException {
	    try {
	        // This enables the java.library.path to be modified at runtime
	        // From a Sun engineer at http://forums.sun.com/thread.jspa?threadID=707176
	        //
	        Field field = ClassLoader.class.getDeclaredField("usr_paths");
	        field.setAccessible(true);
	        String[] paths = (String[])field.get(null);
	        for (int i = 0; i < paths.length; i++) {
	            if (s.equals(paths[i])) {
	                return;
	            }
	        }
	        String[] tmp = new String[paths.length+1];
	        System.arraycopy(paths,0,tmp,0,paths.length);
	        tmp[paths.length] = s;
	        field.set(null,tmp);
	        System.setProperty("java.library.path", System.getProperty("java.library.path") + File.pathSeparator + s);
	    } catch (IllegalAccessException e) {
	        throw new IOException("Failed to get permissions to set library path");
	    } catch (NoSuchFieldException e) {
	        throw new IOException("Failed to get field handle to set library path");
	    }
	}

	
	/**
	 * This function is supposed to allow us to set environmental variables. Taken
	 * from here:
	 * 
	 *   http://stackoverflow.com/questions/318239/how-do-i-set-environment-variables-from-java
	 * 
	 * @param newenv
	 */
	public static void setEnv(Map<String, String> newenv) {
		try {
			Class<?> processEnvironmentClass = Class
					.forName("java.lang.ProcessEnvironment");
			Field theEnvironmentField = processEnvironmentClass
					.getDeclaredField("theEnvironment");
			theEnvironmentField.setAccessible(true);
			Map<String, String> env = (Map<String, String>) theEnvironmentField
					.get(null);
			env.putAll(newenv);
			Field theCaseInsensitiveEnvironmentField = processEnvironmentClass
					.getDeclaredField("theCaseInsensitiveEnvironment");
			theCaseInsensitiveEnvironmentField.setAccessible(true);
			Map<String, String> cienv = (Map<String, String>) theCaseInsensitiveEnvironmentField
					.get(null);
			cienv.putAll(newenv);
		} catch (NoSuchFieldException e) {
			try {
				Class[] classes = Collections.class.getDeclaredClasses();
				Map<String, String> env = System.getenv();
				for (Class cl : classes) {
					if ("java.util.Collections$UnmodifiableMap".equals(cl
							.getName())) {
						Field field = cl.getDeclaredField("m");
						field.setAccessible(true);
						Object obj = field.get(env);
						Map<String, String> map = (Map<String, String>) obj;
						map.clear();
						map.putAll(newenv);
					}
				}
			} catch (Exception e2) {
				e2.printStackTrace();
			}
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

	
	private static void init(String args[]) {
		RLinkInit.args = args;
		RLinkInit.lastError = null;
		 
	}

	public static String getStackTrace(Throwable aThrowable){
		final Writer result = new StringWriter();
		final PrintWriter printWriter = new PrintWriter(result);
		aThrowable.printStackTrace(printWriter);
		return result.toString();
	}
	
	
	private RLinkInit() {		
		rlogger.info("Starting RLinkInit() initialization");
		
		//To prevent REngine from calling exit(1) on error 
		System.setProperty("jri.ignore.ule", "yes");		
		
		try{
			rlogger.info("Setting the native libs location to "+getNativeLibLocation());		
			addDir(getNativeLibLocation()); // Java library path, passed as a first (zero) argument						
		}catch(IOException e){
			//e.printStackTrace();
			rlogger.fatal(getStackTrace(e));
			throw new RLinkException(
					"Unable to set the native libraries directory");
		}		
		boolean versionsMatch = false;
		try{
			//Attempting to load dynamic libs here			
			versionsMatch = Rengine.versionCheck();
		}catch(UnsatisfiedLinkError e){			
			rlogger.fatal(getStackTrace(e));			
			throw new RLinkException("Unable to load dynamic libraries");
		}
				
		if (!versionsMatch) {			
			rlogger.fatal("** Version mismatch - Java files don't match library version.");			
			throw new RLinkException(
					"Version mismatch - Java files don't match library version.");
		}
		
		if(args == null||args.length==0){
			args = new String[]{"--no-save","--verbose"};
		}
	
		rlogger.info("Creating Rengine with arguments: "+args);		
		Rengine re = new Rengine(args, false, new TextConsole());
		
		exec = new RExecutor(re);
		exec.eval(" Sys.setlocale(\"LC_ALL\", \"English\") ");
		rlogger.info("Rengine created, waiting for R");
		
		// the engine creates R is a new thread, so we should wait until it's
		// ready

		if (!re.waitForR()) {
			rlogger.fatal("Cannot load R");			
			throw new RLinkException("Cannot load R");
		}
		
		//Initialize a hash table (R list) for functions
		exec.eval(FUNCTION_HASH_VAR_NAME + " <- list()");
		rlogger.info("R started");
	}

	private static synchronized RLinkInit getInstance() {
		if (init == null) {
			init = new RLinkInit();
		}
		return init;
	}

	public static synchronized boolean installR(String[] args) {
		boolean result = false;
		init(args);
		try{
			result = getRExecutor().isValidRSession();			
		}catch(RLinkException e){
			rlogger.fatal(e.getMessage());
			lastError = e.getMessage();
		}		
		return result;
	}

	
	public static boolean isCurrentValidSession(){
		return init != null && init.exec != null && init.exec.isValidRSession();		
	}
		

	public static synchronized RExecutor getRExecutor() {		
		return getInstance().exec;
	}

	public static synchronized void uninstallR() {
		try{
			if (isCurrentValidSession()) {
				init.exec.stopR();
			}
		}catch(RLinkException e){
			rlogger.fatal(e.getMessage());
			lastError = e.getMessage();
		}
		init.exec = null;
		init = null;
		
	}

}
