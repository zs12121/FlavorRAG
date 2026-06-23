import { useCallback, useRef } from 'react'
import { useAppStore } from '@/store'
import { chatApi, apiUtils } from '@/lib/api'

export const useChat = () => {
  const {
    chat,
    addMessage,
    updateMessage,
    updateMessageMetadata,
    setChatLoading,
    setChatStreaming,
    createChatSession,
    updateSessionTitle,
    addToast
  } = useAppStore()
  
  const abortControllerRef = useRef<AbortController | null>(null)
  
  // 发送消息
  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return
    
    try {
      // 确保有活跃的会话
      let sessionId = chat.currentSession?.id
      if (!sessionId) {
        sessionId = createChatSession()
      }
      
      // 添加用户消息
      addMessage(sessionId, {
        role: 'user',
        content: content.trim()
      })

      // 如果这是第一条消息，更新会话标题
      const currentSession = chat.sessions.find(s => s.id === sessionId)
      if (currentSession && currentSession.messages.length === 0) {
        // 截取前30个字符作为标题，避免标题过长
        const title = content.trim().length > 30
          ? content.trim().substring(0, 30) + '...'
          : content.trim()
        updateSessionTitle(sessionId, title)
      }
      
      // 创建助手消息占位符
      const assistantMessageId = addMessage(sessionId, {
        role: 'assistant',
        content: ''
      })
      
      setChatLoading(true)
      setChatStreaming(true)
      
      // 取消之前的请求
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      
      abortControllerRef.current = new AbortController()
      
      let fullResponse = ''
      let traceId: string | undefined
      
      try {
        // 开始流式响应（通过 onMetadata 回调捕获 trace_id）
        const streamGenerator = chatApi.sendMessage(content, sessionId, (meta) => {
          if (meta.trace_id) traceId = meta.trace_id
        })
        
        for await (const chunk of streamGenerator) {
          fullResponse += chunk
          updateMessage(sessionId, assistantMessageId, fullResponse)
        }
        
        // 如果响应为空，显示错误消息
        if (!fullResponse.trim()) {
          fullResponse = '抱歉，我现在无法回答您的问题。请稍后再试。'
          updateMessage(sessionId, assistantMessageId, fullResponse)
        }
        
        // 流结束后，将 trace_id 写入 message metadata
        if (traceId) {
          updateMessageMetadata(sessionId, assistantMessageId, { trace_id: traceId })
        }
        
      } catch (streamError) {
        console.error('Stream error:', streamError)
        
        // 流式请求失败，尝试普通请求作为后备
        try {
          const fallbackResponse = await chatApi.sendMessage(content, sessionId).next()
          if (fallbackResponse.value) {
            updateMessage(sessionId, assistantMessageId, fallbackResponse.value)
          } else {
            throw new Error('No response from fallback')
          }
        } catch (fallbackError) {
          console.error('Fallback error:', fallbackError)
          updateMessage(
            sessionId,
            assistantMessageId,
            '抱歉，网络连接出现问题。请检查您的网络连接后重试。'
          )
          
          addToast({
            type: 'error',
            title: '发送失败',
            message: '网络连接异常，请重试',
            duration: 5000
          })
        }
      }
      
    } catch (error) {
      console.error('Send message error:', error)
      
      addToast({
        type: 'error',
        title: '发送失败',
        message: apiUtils.handleError(error as any),
        duration: 5000
      })
    } finally {
      setChatLoading(false)
      setChatStreaming(false)
      abortControllerRef.current = null
    }
  }, [
    chat.currentSession?.id,
    chat.sessions,
    addMessage,
    updateMessage,
    updateMessageMetadata,
    setChatLoading,
    setChatStreaming,
    createChatSession,
    updateSessionTitle,
    addToast
  ])
  
  // 停止生成
  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    
    setChatLoading(false)
    setChatStreaming(false)
    
    addToast({
      type: 'info',
      title: '已停止生成',
      duration: 3000
    })
  }, [setChatLoading, setChatStreaming, addToast])
  
  // 重新生成回答
  const regenerateResponse = useCallback(async (messageId: string) => {
    const currentSession = chat.currentSession
    if (!currentSession) return
    
    // 找到要重新生成的消息
    const messageIndex = currentSession.messages.findIndex(m => m.id === messageId)
    if (messageIndex === -1) return
    
    const message = currentSession.messages[messageIndex]
    if (message.role !== 'assistant') return
    
    // 找到对应的用户消息
    const userMessage = currentSession.messages[messageIndex - 1]
    if (!userMessage || userMessage.role !== 'user') return
    
    // 清空助手消息内容和 metadata（trace_id、feedback 等）
    updateMessage(currentSession.id, messageId, '')
    updateMessageMetadata(currentSession.id, messageId, { trace_id: undefined, feedback: undefined })
    
    // 重新发送请求
    setChatLoading(true)
    setChatStreaming(true)
    
    try {
      let fullResponse = ''
      const streamGenerator = chatApi.sendMessage(userMessage.content, currentSession.id)
      
      for await (const chunk of streamGenerator) {
        fullResponse += chunk
        updateMessage(currentSession.id, messageId, fullResponse)
      }
      
    } catch (error) {
      console.error('Regenerate error:', error)
      updateMessage(
        currentSession.id,
        messageId,
        '抱歉，重新生成时出现错误。请稍后再试。'
      )
      
      addToast({
        type: 'error',
        title: '重新生成失败',
        message: apiUtils.handleError(error as any),
        duration: 5000
      })
    } finally {
      setChatLoading(false)
      setChatStreaming(false)
    }
  }, [
    chat.currentSession,
    updateMessage,
    updateMessageMetadata,
    setChatLoading,
    setChatStreaming,
    addToast
  ])
  
  // 复制消息
  const copyMessage = useCallback(async (content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      addToast({
        type: 'success',
        title: '已复制到剪贴板',
        duration: 2000
      })
    } catch (error) {
      console.error('Copy error:', error)
      addToast({
        type: 'error',
        title: '复制失败',
        message: '无法访问剪贴板',
        duration: 3000
      })
    }
  }, [addToast])
  
  // 提供反馈
  const provideFeedback = useCallback(async (messageId: string, type: 'like' | 'dislike' | null) => {
    const currentSession = chat.currentSession
    if (!currentSession) return
    
    // 找到对应消息
    const message = currentSession.messages.find(m => m.id === messageId)
    if (!message) return
    
    const traceId = message.metadata?.trace_id
    
    try {
      // 只有 type 不为 null 时才调用 API（取消反馈不调后端）
      if (type) {
        await chatApi.submitFeedback(messageId, type, currentSession.id, traceId)
      }
      
      // API 成功后才更新 store（避免 API 失败时 store 与 UI 不一致）
      if (type) {
        updateMessageMetadata(currentSession.id, messageId, { feedback: type })
      } else {
        // 取消反馈：清除 metadata 中的 feedback
        updateMessageMetadata(currentSession.id, messageId, { feedback: undefined })
      }
      
      addToast({
        type: 'success',
        title: type === 'like' ? '感谢您的反馈！' : type === 'dislike' ? '我们会继续改进' : '已取消反馈',
        duration: 2000
      })
    } catch (error) {
      console.error('Feedback error:', error)
      addToast({
        type: 'error',
        title: '反馈提交失败',
        duration: 3000
      })
      throw error  // 抛出错误让组件 revert 按钮状态
    }
  }, [chat.currentSession, updateMessageMetadata, addToast])
  
  return {
    // 状态
    currentSession: chat.currentSession,
    sessions: chat.sessions,
    isLoading: chat.isLoading,
    isStreaming: chat.isStreaming,
    
    // 方法
    sendMessage,
    stopGeneration,
    regenerateResponse,
    copyMessage,
    provideFeedback
  }
}