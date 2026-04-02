import React, { useState, useEffect } from "react";
import { toast } from "react-hot-toast";
import {
  X,
  Save,
  LayoutGrid,
  Bot,
  Database,
  Key,
  Globe,
  Cpu,
  Info,
  Loader2,
  CheckCircle2,
  Settings,
  HardDrive,
  FolderOpen,
} from "lucide-react";
import {
  getSettings,
  saveSettings,
  AppSettings,
  checkAiConnection,
  selectFolder,
} from "../services/api";
import { getVersion } from "@tauri-apps/api/app";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type TabKey = "general" | "embedding" | "chat" | "advanced" | "about";

const SettingInput = ({ 
  label, 
  value, 
  onChange, 
  placeholder, 
  type = "text", 
  icon: Icon, 
  options = [] 
}: any) => (
  <div className="form-control">
    <label className="label">
      <span className="label-text font-medium flex items-center gap-2">
        {Icon && <Icon size={14} className="opacity-70" />}
        {label}
      </span>
    </label>
    {options.length > 0 ? (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="select select-bordered select-sm font-mono text-xs"
      >
        {options.map((option: any) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    ) : (
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="input input-bordered input-sm font-mono text-xs"
      />
    )}
  </div>
);

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [config, setConfig] = useState<AppSettings>({
    search_top_k: 50,
    display_density: "comfortable",

    embedding_base_url: "http://localhost:11434/v1",
    embedding_api_key: "ollama",
    embedding_model: "qwen3-embedding:0.6b",

    reranker_type: "dashscope",
    reranker_base_url: "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    reranker_api_key: "sk-b5ed0ddae2bb4ca1ae401a29111e0360",
    reranker_model: "qwen3-vl-rerank",

    enable_ai_chat: false,
    chat_base_url: "http://localhost:11434/v1",
    chat_api_key: "ollama",
    chat_model: "qwen3",
    chat_top_k: 5,
    max_agent_loops: 5,

    use_external_chat_api: false,
    external_chat_base_url: "https://api.minimax.chat/v1",
    external_chat_api_key: "",
    external_chat_model: "MiniMax-M2.5",
    external_chat_api_choice: 1,
    external_chat_base_url_2: "https://longcat.chat/v1",
    external_chat_api_key_2: "",
    external_chat_model_2: "",
  });

  const [activeTab, setActiveTab] = useState<TabKey>("general");
  const [isLoading, setIsLoading] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [appVersion, setAppVersion] = useState("");

  useEffect(() => {
    getVersion().then(setAppVersion);
  }, []);

  useEffect(() => {
    if (isOpen) {
      setIsLoading(true);
      getSettings()
        .then((data) => setConfig((prev) => ({ ...prev, ...data })))
        .catch(() => toast.error("加载设置失败"))
        .finally(() => setIsLoading(false));
    }
  }, [isOpen]);

  const handleSave = async () => {
    try {
      if (config.chat_top_k > config.search_top_k) {
        toast.error("AI参考数量不能大于搜索返回总数");
        return;
      }
      await saveSettings(config);
      toast.success("设置已保存");
      onClose();
    } catch {
      toast.error("保存失败");
    }
  };

  const handleTestConnection = async (type: "embedding" | "chat") => {
    setIsTesting(true);
    const baseUrl =
      type === "embedding" ? config.embedding_base_url : config.chat_base_url;
    const apiKey =
      type === "embedding" ? config.embedding_api_key : config.chat_api_key;
    const model =
      type === "embedding" ? config.embedding_model : config.chat_model;

    toast
      .promise(checkAiConnection(baseUrl, apiKey, model), {
        loading: "正在连接服务器...",
        success: (msg) => msg,
        error: (err) => `测试失败: ${err.message}`,
      })
      .finally(() => setIsTesting(false));
  };

  if (!isOpen) return null;

  return (
    <div className="modal modal-open z-50">
      <div className="modal-backdrop" onClick={onClose}></div>

      <div className="modal-box w-11/12 max-w-4xl h-[650px] p-0 bg-base-100 shadow-2xl border border-base-200 flex flex-col overflow-hidden rounded-xl">
        <div className="flex justify-between items-center px-6 py-4 border-b border-base-200 bg-base-100 shrink-0">
          <h3 className="font-bold text-xl text-base-content">系统设置</h3>
          <button
            onClick={onClose}
            className="btn btn-sm btn-circle btn-ghost hover:bg-base-200"
          >
            <X size={20} />
          </button>
        </div>

        <div className="flex grow overflow-hidden">
          <aside className="w-60 bg-base-200/50 border-r border-base-200 flex flex-col p-3 shrink-0">
            <ul className="menu gap-1 w-full p-0">
              <li>
                <button
                  onClick={() => setActiveTab("general")}
                  className={
                    activeTab === "general"
                      ? "active font-medium"
                      : "font-medium"
                  }
                >
                  <LayoutGrid size={18} />
                  显示与检索
                </button>
              </li>
              <div className="divider my-1 text-xs opacity-50">模型服务</div>
              <li>
                <button
                  onClick={() => setActiveTab("embedding")}
                  className={
                    activeTab === "embedding"
                      ? "active font-medium"
                      : "font-medium"
                  }
                >
                  <Database size={18} />
                  向量模型 (Embedding)
                </button>
              </li>
              <li>
                <button
                  onClick={() => setActiveTab("chat")}
                  className={
                    activeTab === "chat" ? "active font-medium" : "font-medium"
                  }
                >
                  <Bot size={18} />
                  AI 问答 (Chat)
                </button>
              </li>
              <div className="divider my-1 text-xs opacity-50">高级</div>
              <li>
                <button
                  onClick={() => setActiveTab("advanced")}
                  className={
                    activeTab === "advanced"
                      ? "active font-medium"
                      : "font-medium"
                  }
                >
                  <Settings size={18} />
                  高级设置
                </button>
              </li>
              <div className="divider my-1 text-xs opacity-50">其他</div>
              <li>
                <button
                  onClick={() => setActiveTab("about")}
                  className={
                    activeTab === "about" ? "active font-medium" : "font-medium"
                  }
                >
                  <Info size={18} />
                  关于软件
                </button>
              </li>
            </ul>
          </aside>

          <main className="grow overflow-y-auto p-8 bg-base-100">
            {activeTab === "general" && (
              <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
                <div>
                  <h3 className="font-bold text-lg mb-1">基础设置</h3>
                  <p className="text-xs text-base-content/60">
                    调整界面的显示方式和搜索结果数量。
                  </p>
                </div>

                <fieldset className="fieldset bg-base-200/30 p-4 rounded-box border border-base-200">
                  <legend className="fieldset-legend font-bold">
                    搜索数量 (Top K)
                  </legend>
                  <div className="flex items-center gap-4 mt-2">
                    <input
                      type="range"
                      min="5"
                      step="5"
                      value={config.search_top_k}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          search_top_k: parseInt(e.target.value),
                        })
                      }
                      className="range range-primary range-sm"
                    />
                    <span className="badge badge-lg font-mono">
                      {config.search_top_k}
                    </span>
                  </div>
                </fieldset>

                <fieldset className="fieldset bg-base-200/30 p-4 rounded-box border border-base-200">
                  <legend className="fieldset-legend font-bold">
                    显示密度
                  </legend>
                  <div className="join mt-2">
                    <input
                      className="join-item btn btn-sm"
                      type="radio"
                      name="density"
                      aria-label="舒适"
                      checked={config.display_density === "comfortable"}
                      onChange={() =>
                        setConfig({ ...config, display_density: "comfortable" })
                      }
                    />
                    <input
                      className="join-item btn btn-sm"
                      type="radio"
                      name="density"
                      aria-label="紧凑"
                      checked={config.display_density === "compact"}
                      onChange={() =>
                        setConfig({ ...config, display_density: "compact" })
                      }
                    />
                  </div>
                </fieldset>
              </div>
            )}

            {activeTab === "embedding" && (
              <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-bold text-lg mb-1">向量模型服务</h3>
                    <p className="text-xs text-base-content/60">
                      用于将搜索词转换为向量。
                    </p>
                  </div>
                  <button
                    onClick={() => handleTestConnection("embedding")}
                    className="btn btn-sm btn-neutral gap-2"
                    disabled={isTesting}
                  >
                    {isTesting ? (
                      <Loader2 className="animate-spin" size={14} />
                    ) : (
                      <CheckCircle2 size={14} />
                    )}
                    测试连接
                  </button>
                </div>

                <div className="bg-base-200/30 p-6 rounded-xl border border-base-200 space-y-4">
                  <SettingInput
                    label="API 地址"
                    icon={Globe}
                    value={config.embedding_base_url}
                    onChange={(v: string) =>
                      setConfig({ ...config, embedding_base_url: v })
                    }
                    placeholder="http://localhost:11434/v1"
                  />
                  <SettingInput
                    label="API 密钥"
                    icon={Key}
                    type="password"
                    value={config.embedding_api_key}
                    onChange={(v: string) =>
                      setConfig({ ...config, embedding_api_key: v })
                    }
                    placeholder="ollama (或 OpenAI Key)"
                  />
                  <SettingInput
                    label="模型名称"
                    icon={Cpu}
                    value={config.embedding_model}
                    onChange={(v: string) =>
                      setConfig({ ...config, embedding_model: v })
                    }
                    placeholder="qwen3-embedding:0.6b"
                  />
                </div>
                <div className="alert alert-warning text-xs">
                  警告：本项目基于 qwen3-embedding:0.6b
                  模型制作，如果更换模型可能会导致结果不正确。
                </div>
              </div>
            )}

            {activeTab === "chat" && (
              <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
                <div>
                  <h3 className="font-bold text-lg mb-1">AI 问答助手</h3>
                  <p className="text-xs text-base-content/60">
                    用于基于搜索结果回答用户提问 (RAG)。
                  </p>
                </div>

                <div className="form-control">
                  <label className="label cursor-pointer justify-start gap-4 border border-base-200 p-3 rounded-lg hover:bg-base-200/30">
                    <input
                      type="checkbox"
                      className="toggle toggle-success"
                      checked={config.enable_ai_chat}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          enable_ai_chat: e.target.checked,
                        })
                      }
                    />
                    <span className="font-bold">启用 AI 问答</span>
                  </label>
                </div>

                {config.enable_ai_chat && (
                  <div className="bg-base-200/30 p-6 rounded-xl border border-base-200 space-y-4 animate-in fade-in slide-in-from-top-2">
                    <div className="flex justify-end">
                      <button
                        onClick={() => handleTestConnection("chat")}
                        className="btn btn-xs btn-outline gap-1"
                        disabled={isTesting}
                      >
                        {isTesting ? (
                          <Loader2 className="animate-spin" size={12} />
                        ) : (
                          <CheckCircle2 size={12} />
                        )}
                        测试连接
                      </button>
                    </div>
                    <SettingInput
                      label="API 地址"
                      icon={Globe}
                      value={config.chat_base_url}
                      onChange={(v: string) =>
                        setConfig({ ...config, chat_base_url: v })
                      }
                      placeholder="http://localhost:11434/v1"
                    />
                    <SettingInput
                      label="API 密钥"
                      icon={Key}
                      type="password"
                      value={config.chat_api_key}
                      onChange={(v: string) =>
                        setConfig({ ...config, chat_api_key: v })
                      }
                      placeholder="ollama (或 OpenAI Key)"
                    />
                    <SettingInput
                      label="模型名称"
                      icon={Cpu}
                      value={config.chat_model}
                      onChange={(v: string) =>
                        setConfig({ ...config, chat_model: v })
                      }
                      placeholder="qwen3:7b"
                    />

                    <div className="divider"></div>

                    <div className="form-control">
                      <label className="label">
                        <span className="label-text font-medium">
                          参考条文数量 ({config.chat_top_k})
                        </span>
                      </label>
                      <input
                        type="range"
                        min="1"
                        step="1"
                        value={config.chat_top_k}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            chat_top_k: parseInt(e.target.value),
                          })
                        }
                        className="range range-success range-xs"
                      />
                      <label className="label">
                        <span className="label-text-alt text-base-content/50">
                          传递给 AI 的上下文数量，过多可能会导致幻觉。
                        </span>
                      </label>
                    </div>
                    <div className="form-control mt-4">
                      <label className="label">
                        <span className="label-text font-medium flex items-center gap-2">
                          <Bot size={14} className="opacity-70" />
                          智能体最大思考轮数
                        </span>
                        <span className="label-text-alt">
                          {config.max_agent_loops <= 0
                            ? "∞ (自动托管)"
                            : `${config.max_agent_loops} 次`}
                        </span>
                      </label>
                      <div className="flex items-center gap-4">
                        <input
                          type="range"
                          min="0"
                          max="30"
                          step="1"
                          value={config.max_agent_loops}
                          onChange={(e) =>
                            setConfig({
                              ...config,
                              max_agent_loops: parseInt(e.target.value),
                            })
                          }
                          className="range range-primary range-xs flex-1"
                        />
                        <div
                          className="tooltip"
                          data-tip="设置为 0 表示由AI自动判断结束时机 (上限99轮)"
                        >
                          <input
                            type="number"
                            className="input input-bordered input-sm w-20 text-center"
                            value={config.max_agent_loops}
                            onChange={(e) =>
                              setConfig({
                                ...config,
                                max_agent_loops: parseInt(e.target.value) || 0,
                              })
                            }
                          />
                        </div>
                      </div>
                      <label className="label">
                        <span className="label-text-alt text-base-content/50">
                          允许 AI 自动修正计划并重新搜索的最大次数。推荐 3-5
                          次。设为 0 时，AI 会一直搜索直到它认为信息充足（适合复杂案件）。
                        </span>
                      </label>
                    </div>
                  </div>
                )}

                {/* Reranker 配置 */}
                <div className="divider"></div>

                <div>
                  <h3 className="font-bold text-lg mb-1">重排序模型 (Reranker)</h3>
                  <p className="text-xs text-base-content/60">
                    配置搜索结果的重排序模型，提升搜索相关性。
                  </p>
                </div>

                <div className="form-control">
                  <label className="label cursor-pointer justify-start gap-4 border border-base-200 p-3 rounded-lg hover:bg-base-200/30">
                    <input
                      type="checkbox"
                      className="toggle toggle-primary"
                      checked={config.reranker_type !== "local"}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          reranker_type: e.target.checked ? "dashscope" : "local",
                        })
                      }
                    />
                    <span className="font-bold">启用 Reranker 重排序</span>
                  </label>
                </div>

                {config.reranker_type !== "local" && (
                  <div className="bg-base-200/30 p-6 rounded-xl border border-base-200 space-y-4 animate-in fade-in slide-in-from-top-2">
                    <SettingInput
                      label="Reranker 类型"
                      icon={Settings}
                      value={config.reranker_type}
                      onChange={(v: string) =>
                        setConfig({ ...config, reranker_type: v as "dashscope" | "local" | "custom" })
                      }
                      placeholder="dashscope"
                      type="select"
                      options={[
                        { value: "dashscope", label: "阿里云 DashScope" },
                        { value: "custom", label: "自定义 API" }
                      ]}
                    />
                    <SettingInput
                      label="API 地址"
                      icon={Globe}
                      value={config.reranker_base_url}
                      onChange={(v: string) =>
                        setConfig({ ...config, reranker_base_url: v })
                      }
                      placeholder={config.reranker_type === "dashscope" ? "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank" : "https://api.example.com/rerank"}
                    />
                    <SettingInput
                      label="API 密钥"
                      icon={Key}
                      type="password"
                      value={config.reranker_api_key}
                      onChange={(v: string) =>
                        setConfig({ ...config, reranker_api_key: v })
                      }
                      placeholder="sk-..."
                    />
                    <SettingInput
                      label="模型名称"
                      icon={Cpu}
                      value={config.reranker_model}
                      onChange={(v: string) =>
                        setConfig({ ...config, reranker_model: v })
                      }
                      placeholder={config.reranker_type === "dashscope" ? "qwen3-vl-rerank" : "your-rerank-model"}
                    />
                  </div>
                )}
              </div>
            )}
            {activeTab === "advanced" && (
              <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
                <div>
                  <h3 className="font-bold text-lg mb-1">高级设置</h3>
                  <p className="text-xs text-base-content/60">
                    如无必要，请勿调整。
                  </p>
                </div>

                <fieldset className="fieldset bg-base-200/30 p-4 rounded-box border border-base-200 border-l-4 border-l-secondary">
                  <legend className="fieldset-legend font-bold flex items-center gap-2">
                    <HardDrive size={14} /> 数据库位置 (高级)
                  </legend>

                  <div className="flex flex-col gap-2 mt-1 w-full">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        className="input input-sm input-bordered flex-1 font-mono text-xs opacity-80"
                        value={config.custom_data_path || ""}
                        placeholder="默认 (程序内部路径)"
                        readOnly
                      />
                      <button
                        className="btn btn-sm btn-secondary"
                        onClick={async () => {
                          const path = await selectFolder();
                          if (path) {
                            setConfig({ ...config, custom_data_path: path });
                            toast.success(
                              "路径已选择，请确保该目录下包含 content.db 和 law_db.lancedb"
                            );
                          }
                        }}
                      >
                        <FolderOpen size={16} /> 更改
                      </button>
                      {config.custom_data_path && (
                        <button
                          className="btn btn-sm btn-ghost text-error"
                          onClick={() =>
                            setConfig({ ...config, custom_data_path: "" })
                          }
                          title="恢复默认"
                        >
                          <X size={16} />
                        </button>
                      )}
                    </div>
                    <p className="text-[10px] text-base-content/50 leading-tight">
                      可以将数据库放在 NAS 或共享文件夹中供团队使用。
                      <br />
                      <span className="text-warning">
                        注意：更改后，请手动将 <b>content.db</b> 和{" "}
                        <b>law_db.lancedb</b>{" "}
                        文件夹移动到新路径下，否则无法搜索。
                      </span>
                    </p>
                  </div>
                </fieldset>
              </div>
            )}

            {activeTab === "about" && (
              <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300 text-center md:text-left">
                <div className="flex flex-col items-center md:items-start gap-4">
                  <div className="bg-primary/10 p-4 rounded-2xl">
                    <Key size={48} className="text-primary" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-extrabold text-base-content">
                      API 配置
                    </h2>
                    <p className="text-sm text-base-content/60 font-mono mt-1">
                      LawVault v{appVersion}
                    </p>
                  </div>
                  <p className="text-base-content/80 leading-relaxed max-w-prose">
                    启用外部云端 API 可替代本地模型，无需运行 Ollama 服务。
                    关闭开关则使用本地 Ollama 模型。
                  </p>
                </div>

                <div className="divider"></div>

                <div className="form-control">
                  <label className="label cursor-pointer justify-start gap-4">
                    <input
                      type="checkbox"
                      className="toggle toggle-primary"
                      checked={config.use_external_chat_api}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          use_external_chat_api: e.target.checked,
                        }))
                      }
                    />
                    <span className="label-text font-semibold text-base">
                      启用外部 API
                    </span>
                  </label>
                  <p className="text-xs text-base-content/50 mt-1 ml-1">
                    开启后，AI 助手将使用云端模型，不再依赖本地 Ollama
                  </p>
                </div>

                {config.use_external_chat_api && (
                  <>
                    <div className="form-control">
                      <label className="label">
                        <span className="label-text font-medium">选择云端 API</span>
                      </label>
                      <select
                        className="select select-bordered select-sm"
                        value={config.external_chat_api_choice}
                        onChange={(e) =>
                          setConfig((prev) => ({
                            ...prev,
                            external_chat_api_choice: parseInt(e.target.value),
                          }))
                        }
                      >
                        <option value={1}>MiniMax</option>
                        <option value={2}>LongCat</option>
                      </select>
                    </div>

                    {config.external_chat_api_choice === 1 && (
                      <div className="card bg-base-200/50 border border-base-200">
                        <div className="card-body p-5">
                          <h3 className="card-title text-base mb-4 flex items-center gap-2">
                            <Globe size={16} className="text-primary" />
                            MiniMax API
                          </h3>
                          <div className="grid grid-cols-1 gap-4">
                            <SettingInput
                              label="Base URL"
                              value={config.external_chat_base_url}
                              onChange={(val: string) =>
                                setConfig((prev) => ({
                                  ...prev,
                                  external_chat_base_url: val,
                                }))
                              }
                              placeholder="https://api.minimax.chat/v1"
                              icon={Globe}
                            />
                            <SettingInput
                              label="API Key"
                              value={config.external_chat_api_key}
                              onChange={(val: string) =>
                                setConfig((prev) => ({
                                  ...prev,
                                  external_chat_api_key: val,
                                }))
                              }
                              placeholder="sk-..."
                              icon={Key}
                            />
                            <SettingInput
                              label="模型"
                              value={config.external_chat_model}
                              onChange={(val: string) =>
                                setConfig((prev) => ({
                                  ...prev,
                                  external_chat_model: val,
                                }))
                              }
                              placeholder="MiniMax-M2.5"
                              icon={Cpu}
                            />
                          </div>
                          <div className="mt-4">
                            <button
                              onClick={async () => {
                                setIsTesting(true);
                                try {
                                  const result = await checkAiConnection(
                                    config.external_chat_base_url,
                                    config.external_chat_api_key,
                                    config.external_chat_model
                                  );
                                  toast.success(result);
                                } catch (e: any) {
                                  toast.error(`测试失败: ${e.message}`);
                                } finally {
                                  setIsTesting(false);
                                }
                              }}
                              className="btn btn-sm btn-neutral gap-2"
                              disabled={isTesting}
                            >
                              {isTesting ? (
                                <Loader2 className="animate-spin" size={14} />
                              ) : (
                                <CheckCircle2 size={14} />
                              )}
                              测试连接
                            </button>
                          </div>
                        </div>
                      </div>
                    )}

                    {config.external_chat_api_choice === 2 && (
                      <div className="card bg-base-200/50 border border-base-200">
                        <div className="card-body p-5">
                          <h3 className="card-title text-base mb-4 flex items-center gap-2">
                            <Globe size={16} className="text-primary" />
                            LongCat API
                          </h3>
                          <div className="grid grid-cols-1 gap-4">
                            <SettingInput
                              label="Base URL"
                              value={config.external_chat_base_url_2}
                              onChange={(val: string) =>
                                setConfig((prev) => ({
                                  ...prev,
                                  external_chat_base_url_2: val,
                                }))
                              }
                              placeholder="https://longcat.chat/v1"
                              icon={Globe}
                            />
                            <SettingInput
                              label="API Key"
                              value={config.external_chat_api_key_2}
                              onChange={(val: string) =>
                                setConfig((prev) => ({
                                  ...prev,
                                  external_chat_api_key_2: val,
                                }))
                              }
                              placeholder="sk-..."
                              icon={Key}
                            />
                            <SettingInput
                              label="模型"
                              value={config.external_chat_model_2}
                              onChange={(val: string) =>
                                setConfig((prev) => ({
                                  ...prev,
                                  external_chat_model_2: val,
                                }))
                              }
                              placeholder="longcat-model"
                              icon={Cpu}
                            />
                          </div>
                          <div className="mt-4">
                            <button
                              onClick={async () => {
                                setIsTesting(true);
                                try {
                                  const result = await checkAiConnection(
                                    config.external_chat_base_url_2,
                                    config.external_chat_api_key_2,
                                    config.external_chat_model_2
                                  );
                                  toast.success(result);
                                } catch (e: any) {
                                  toast.error(`测试失败: ${e.message}`);
                                } finally {
                                  setIsTesting(false);
                                }
                              }}
                              className="btn btn-sm btn-neutral gap-2"
                              disabled={isTesting}
                            >
                              {isTesting ? (
                                <Loader2 className="animate-spin" size={14} />
                              ) : (
                                <CheckCircle2 size={14} />
                              )}
                              测试连接
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </main>
        </div>

        <div className="p-4 border-t border-base-200 bg-base-100 flex justify-end gap-2 shrink-0">
          <button
            className="btn btn-ghost"
            onClick={onClose}
            disabled={isLoading}
          >
            取消
          </button>
          <button
            className="btn btn-primary gap-2 px-6"
            onClick={handleSave}
            disabled={isLoading}
          >
            {isLoading ? (
              <span className="loading loading-spinner loading-xs"></span>
            ) : (
              <Save size={18} />
            )}
            保存配置
          </button>
        </div>
      </div>
    </div>
  );
};
